"""
PPO (stable-baselines3) on BusPriorityEnv, 4 parallel envs, with validation.

    python bo4_bus/rl/train_ppo.py --reward vehicle --steps 200000      # RL-D
    python bo4_bus/rl/train_ppo.py --reward person  --steps 200000      # RL-E
    tensorboard --logdir bo4_bus/rl/runs

Training draws SUMO seeds 0-899 only (env.TRAIN_SEEDS). Every --eval-every steps (and
before the first update) the deterministic policy is evaluated on validation seeds
900-904 with the BO4 measures (evaluate.run_one: the policy at B1 inside the grid
runner, C elsewhere, warm-up 600 s + --val-duration s). The best checkpoint is kept by
    reward D (vehicle): veh_delay_B1       reward E (person): person_delay_B1_h
(mean over the 5 validation seeds). C at B1 on the same seeds is evaluated once as a
reference. Outputs in bo4_bus/rl/runs/<name>/ (gitignored):
    tb/                      TensorBoard (SB3 train/* and rollout/*, plus val/*)
    validation.csv           one row per (checkpoint, seed), written as soon as it finishes
    validation_summary.csv   one row per checkpoint (means over the validation seeds)
    ckpt_<steps>.zip, best_model.zip, best.json, final_model.zip, monitor.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as ev  # noqa: E402
from env import BusPriorityEnv  # noqa: E402
from c_experiments import _load, _save  # noqa: E402

SELECT = {"vehicle": "veh_delay_B1", "person": "person_delay_B1_h"}
VAL_COLS = ["bus_delay_B1", "veh_delay_B1", "person_delay_B1_h", "bunch_last_total",
            "car_delay_total", "person_delay_h", "bus_jdelay", "cross_delay",
            "rl_shield_overrides", "rl_violations", "never_inserted", "teleports"]


def make_env(reward, rank):
    def _init():
        return BusPriorityEnv(reward=reward)
    return _init


def validate(out, name, steps, policy, val_duration):
    """Run the validation seeds, append each row to validation.csv as soon as it finishes."""
    path = os.path.join(out, "validation.csv")
    rows = _load(path)
    for seed in ev.VAL_SEEDS:
        if any(r["name"] == name and int(r["steps"]) == steps and int(r["seed"]) == seed
               for r in rows):
            continue                                         # already done (restart)
        row, _ = ev.run_one(name, seed, policy=policy, duration=val_duration,
                            tag=f"rl_val_{os.path.basename(out)}")
        assert row["backend"] == "libsumo"
        rows.append({"steps": steps, **{k: row[k] for k in ["name", "seed", "wall_s"] + VAL_COLS}})
        _save(rows, path)
    cur = pd.DataFrame([r for r in rows if r["name"] == name and int(r["steps"]) == steps])
    return cur[VAL_COLS].mean().to_dict()


def make_callback(out, reward, every, val_duration):
    from stable_baselines3.common.callbacks import BaseCallback

    class ValidationCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = 0
            self.best = None
            self.summary_path = os.path.join(out, "validation_summary.csv")
            self.summary = _load(self.summary_path)

        def _evaluate(self):
            steps = int(self.num_timesteps)
            model = self.model
            policy = lambda obs: int(model.predict(obs, deterministic=True)[0])  # noqa: E731
            tic = time.time()
            m = validate(out, "RL", steps, policy, val_duration)
            path = os.path.join(out, f"ckpt_{steps}.zip")
            model.save(path)
            key = SELECT[reward]
            is_best = self.best is None or m[key] < self.best[key]
            if is_best:
                self.best = {"steps": steps, key: m[key], **m}
                model.save(os.path.join(out, "best_model.zip"))
                with open(os.path.join(out, "best.json"), "w") as f:
                    json.dump(self.best, f, indent=1)
            for k, v in m.items():
                self.logger.record(f"val/{k}", v)
            self.logger.record("val/is_best", float(is_best))
            self.logger.dump(steps)
            self.summary.append({"steps": steps, "best": is_best,
                                 "val_wall_s": round(time.time() - tic, 1), **m})
            _save_summary(self.summary, self.summary_path)
            print(f"[val] {steps} steps: {key} = {m[key]:.3f} (best {self.best[key]:.3f} "
                  f"@ {self.best['steps']}), bunching {m['bunch_last_total']:.1f}, "
                  f"violations {m['rl_violations']:.0f}", flush=True)
            self.next_eval = steps + every

        def _on_training_start(self):
            self._evaluate()                                  # untrained policy

        def _on_step(self):
            if self.num_timesteps >= self.next_eval:
                self._evaluate()
            return True

        def _on_training_end(self):
            if self.summary and int(self.summary[-1]["steps"]) != int(self.num_timesteps):
                self._evaluate()

    return ValidationCallback()


def _save_summary(rows, path):
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def reference_c(out, val_duration):
    """C at B1 on the validation seeds (once per run folder)."""
    m = validate(out, "C_all", 0, None, val_duration)
    print("[val] C at B1 reference: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="vehicle", choices=["vehicle", "person"])
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    ap.add_argument("--eval-every", type=int, default=20000)
    ap.add_argument("--val-duration", type=float, default=3600.0,
                    help="validation measurement window after the 600 s warm-up [s]")
    ap.add_argument("--seed", type=int, default=0, help="PPO seed (and training-seed draws)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    torch.set_num_threads(2)

    name = a.name or f"ppo_{'D' if a.reward == 'vehicle' else 'E'}_{a.steps // 1000}k"
    out = os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    reference_c(out, a.val_duration)
    vec = VecMonitor(SubprocVecEnv([make_env(a.reward, i) for i in range(a.n_envs)],
                                   start_method="spawn"),
                     os.path.join(out, "monitor.csv"))
    model = PPO("MlpPolicy", vec, n_steps=a.n_steps, batch_size=64, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                seed=a.seed, verbose=0, device="cpu", tensorboard_log=os.path.join(out, "tb"))
    tic = time.time()
    model.learn(total_timesteps=a.steps, callback=make_callback(out, a.reward, a.eval_every,
                                                                a.val_duration),
                tb_log_name="ppo")
    model.save(os.path.join(out, "final_model.zip"))
    vec.close()
    print(f"trained {model.num_timesteps} steps in {(time.time() - tic) / 60:.1f} min; "
          f"best checkpoint in {out}/best.json", flush=True)


if __name__ == "__main__":
    main()
