"""
PPO (stable-baselines3) on BusPriorityEnv, 4 parallel envs, with validation.

    python bo4_bus/rl/train_ppo.py --reward vehicle --steps 200000      # RL-D
    python bo4_bus/rl/train_ppo.py --reward person  --steps 200000      # RL-E
    python bo4_bus/rl/train_ppo.py --reward regular --w 1 --obs v2 --train-seed 0   # RL-F
    tensorboard --logdir bo4_bus/rl/runs

Run folder (unless --name): runs/ppo_<D|E|F>[_w<w>]_<obs>_s<train seed>_<steps/1000>k,
e.g. runs/ppo_F_w1_v2_s0_200k; config.json holds every option. (The older runs
ppo_D_200k / ppo_E_200k are v1, train seed 0.)

--train-seed seeds PPO (torch, numpy, random) and the 4 envs' generators (SB3 gives env
i the seed train_seed + i). It is NOT a SUMO seed: each episode the env draws its SUMO
seed from 0-899 with that generator, so a run is reproducible for a given train seed.

Training draws SUMO seeds 0-899 only (env.TRAIN_SEEDS). Every --eval-every steps (and
before the first update) the deterministic policy is evaluated on validation seeds
900-904 with the BO4 measures (evaluate.run_one: the policy at B1 inside the grid
runner, C elsewhere, warm-up 600 s + --val-duration s). The best checkpoint is kept by
    reward D (vehicle): veh_delay_B1       reward E (person) and F (regular): person_delay_B1_h
(mean over the 5 validation seeds). C at B1 on the same seeds is evaluated once as a
reference. Reward F (regular) also keeps best_regular.zip: the lowest bunch_last_total
among the checkpoints whose person_delay_B1_h is not worse than C_all's (ties: lower
person_delay_B1_h; see better_regular). Compare the w values with evaluate.py val.
Outputs in bo4_bus/rl/runs/<name>/ (gitignored):
    tb/                      TensorBoard (SB3 train/* and rollout/*, plus val/*)
    validation.csv           one row per (checkpoint, seed), written as soon as it finishes
    validation_summary.csv   one row per checkpoint (means over the validation seeds)
    ckpt_<steps>.zip, best_model.zip, best.json, final_model.zip, monitor.csv
    best_regular.zip         reward F only; best.json then also holds "best_regular"
                             (None until a checkpoint qualifies) and "c_all_reference"
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
import observation as ob  # noqa: E402
from env import TRAIN_SEEDS, BusPriorityEnv  # noqa: E402
from c_experiments import _load, _save  # noqa: E402

SELECT = {"vehicle": "veh_delay_B1", "person": "person_delay_B1_h",
          "regular": "person_delay_B1_h"}
LETTER = {"vehicle": "D", "person": "E", "regular": "F"}
VAL_COLS = ["bus_delay_B1", "veh_delay_B1", "person_delay_B1_h", "bunch_last_total",
            "cv_last_mean", "car_delay_total", "person_delay_h", "bus_jdelay", "cross_delay",
            "rl_shield_overrides", "rl_violations", "never_inserted", "teleports"]


def make_env(reward, rank, w=1.0, obs="v1"):
    def _init():
        return BusPriorityEnv(reward=reward, w=w, obs=obs)
    return _init


def run_name(reward, w, obs, train_seed, steps):
    """runs/ folder name, e.g. ppo_F_w1_v2_s0_200k (w only for reward F)."""
    w_part = f"_w{w:g}" if reward == "regular" else ""
    return f"ppo_{LETTER[reward]}{w_part}_{obs}_s{train_seed}_{steps // 1000}k"


def validate(out, name, steps, policy, val_duration, obs="v1"):
    """Run the validation seeds, append each row to validation.csv as soon as it finishes."""
    path = os.path.join(out, "validation.csv")
    rows = _load(path)
    for seed in ev.VAL_SEEDS:
        if any(r["name"] == name and int(r["steps"]) == steps and int(r["seed"]) == seed
               for r in rows):
            continue                                         # already done (restart)
        row, _ = ev.run_one(name, seed, policy=policy, duration=val_duration,
                            tag=f"rl_val_{os.path.basename(out)}", obs_version=obs)
        assert row["backend"] == "libsumo"
        rows.append({"steps": steps, **{k: row[k] for k in ["name", "seed", "wall_s"] + VAL_COLS}})
        _save(rows, path)
    cur = pd.DataFrame([r for r in rows if r["name"] == name and int(r["steps"]) == steps])
    return cur[VAL_COLS].mean().to_dict()


REGULAR_KEY, REGULAR_GUARD = "bunch_last_total", "person_delay_B1_h"


def better_regular(m, best, c_ref):
    """Reward F's second checkpoint: is validation result `m` better than `best` (None
    if no checkpoint qualified yet)? Only checkpoints whose person delay at B1 is not
    worse than C_all's qualify; then lowest bunching, ties broken by person delay."""
    if m[REGULAR_GUARD] > c_ref[REGULAR_GUARD]:
        return False
    return best is None or ((m[REGULAR_KEY], m[REGULAR_GUARD])
                            < (best[REGULAR_KEY], best[REGULAR_GUARD]))


def make_callback(out, reward, every, val_duration, obs="v1", c_ref=None):
    """c_ref: C_all's validation means (reference_c), needed for reward "regular"."""
    from stable_baselines3.common.callbacks import BaseCallback
    regular = reward == "regular"
    assert not regular or c_ref is not None

    class ValidationCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = 0
            self.best = None
            self.best_regular = None
            self.summary_path = os.path.join(out, "validation_summary.csv")
            self.summary = _load(self.summary_path)

        def _evaluate(self):
            steps = int(self.num_timesteps)
            model = self.model
            policy = lambda obs: int(model.predict(obs, deterministic=True)[0])  # noqa: E731
            tic = time.time()
            m = validate(out, "RL", steps, policy, val_duration, obs)
            path = os.path.join(out, f"ckpt_{steps}.zip")
            model.save(path)
            key = SELECT[reward]
            is_best = self.best is None or m[key] < self.best[key]
            if is_best:
                self.best = {"steps": steps, key: m[key], **m}
                model.save(os.path.join(out, "best_model.zip"))
            is_best_regular = regular and better_regular(m, self.best_regular, c_ref)
            if is_best_regular:
                self.best_regular = {"steps": steps, REGULAR_KEY: m[REGULAR_KEY], **m}
                model.save(os.path.join(out, "best_regular.zip"))
            if is_best or is_best_regular:
                self._write_best()
            for k, v in m.items():
                self.logger.record(f"val/{k}", v)
            self.logger.record("val/is_best", float(is_best))
            extra = {}
            if regular:
                self.logger.record("val/is_best_regular", float(is_best_regular))
                extra = {"best_regular": is_best_regular}
            self.logger.dump(steps)
            self.summary.append({"steps": steps, "best": is_best, **extra,
                                 "val_wall_s": round(time.time() - tic, 1), **m})
            _save_summary(self.summary, self.summary_path)
            msg = (f"[val] {steps} steps: {key} = {m[key]:.3f} (best {self.best[key]:.3f} "
                   f"@ {self.best['steps']}), bunching {m['bunch_last_total']:.1f}, "
                   f"violations {m['rl_violations']:.0f}")
            if regular:
                br = self.best_regular
                msg += ("; best_regular " +(f"bunching {br[REGULAR_KEY]:.1f} @ {br['steps']}"
                                               if br else "none yet (no checkpoint within "
                                               f"C_all's {REGULAR_GUARD} {c_ref[REGULAR_GUARD]:.3f})"))
            print(msg, flush=True)
            self.next_eval = steps + every

        def _write_best(self):
            """best.json: best_model.zip's validation means at the top level (as for D/E);
            for reward F also best_regular.zip's (None if no checkpoint qualified) and the
            C_all reference it was checked against."""
            best = dict(self.best)
            if regular:
                best.update(best_regular=self.best_regular,
                            best_regular_rule=f"lowest {REGULAR_KEY} (ties: {REGULAR_GUARD}) "
                                              f"among checkpoints with {REGULAR_GUARD} <= C_all's",
                            c_all_reference=c_ref)
            with open(os.path.join(out, "best.json"), "w") as f:
                json.dump(best, f, indent=1)

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
    ap.add_argument("--reward", default="vehicle", choices=["vehicle", "person", "regular"],
                    help="vehicle = D, person = E, regular = F")
    ap.add_argument("--w", type=float, default=1.0, help="lateness weight of reward F")
    ap.add_argument("--obs", default="v1", choices=["v1", "v2"], help="observation version")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    ap.add_argument("--eval-every", type=int, default=20000)
    ap.add_argument("--val-duration", type=float, default=3600.0,
                    help="validation measurement window after the 600 s warm-up [s]")
    ap.add_argument("--train-seed", "--seed", dest="train_seed", type=int, default=0,
                    help="seed of PPO / torch and of the envs' SUMO-seed draws (not a SUMO seed)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.reward != "regular" and a.w != 1.0:
        ap.error("--w only applies to --reward regular")

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    torch.set_num_threads(2)

    name = a.name or run_name(a.reward, a.w, a.obs, a.train_seed, a.steps)
    out = os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    cfg = {**vars(a), "name": name, "agent": LETTER[a.reward],
           "w": a.w if a.reward == "regular" else None, "obs_size": int(ob.bounds(a.obs)[0].size),
           "select": SELECT[a.reward], "train_sumo_seeds": list(TRAIN_SEEDS),
           "val_seeds": ev.VAL_SEEDS}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    c_ref = reference_c(out, a.val_duration)
    vec = VecMonitor(SubprocVecEnv([make_env(a.reward, i, a.w, a.obs) for i in range(a.n_envs)],
                                   start_method="spawn"),
                     os.path.join(out, "monitor.csv"))
    model = PPO("MlpPolicy", vec, n_steps=a.n_steps, batch_size=64, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                seed=a.train_seed, verbose=0, device="cpu",
                tensorboard_log=os.path.join(out, "tb"))
    tic = time.time()
    model.learn(total_timesteps=a.steps, callback=make_callback(out, a.reward, a.eval_every,
                                                                a.val_duration, a.obs, c_ref),
                tb_log_name="ppo")
    model.save(os.path.join(out, "final_model.zip"))
    vec.close()
    print(f"trained {model.num_timesteps} steps in {(time.time() - tic) / 60:.1f} min; "
          f"best checkpoint in {out}/best.json", flush=True)


if __name__ == "__main__":
    main()
