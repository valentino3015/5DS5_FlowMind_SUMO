"""
PPO (stable-baselines3) on BusPriorityEnv, 4 parallel envs, with validation.

    python bo4_bus/rl/train_ppo.py --reward vehicle --steps 200000      # RL-D
    python bo4_bus/rl/train_ppo.py --reward person  --steps 200000      # RL-E
    python bo4_bus/rl/train_ppo.py --reward regular --w 1 --obs v2 --train-seed 0   # RL-F
    python bo4_bus/rl/train_ppo.py --reward regular_g --w 5 --w-early 1 --c-switch 0.5 \
                                   --obs v2 --train-seed 0                         # RL-G
    tensorboard --logdir bo4_bus/rl/runs

Agent G (--reward regular_g) differs from F in four ways (all defaults for regular_g):
    reward   F's reward + early-bus penalty (--w-early) + switch cost (--c-switch), reward.py
    --mask   MaskablePPO (sb3-contrib) with env.action_masks(): actions the shield would
             override are never sampled; the shield stays as the final check
    --val-all9  every checkpoint is also validated at all 9 junctions (evaluate.py
             --scope all9) on 900-904; validation runs in --val-workers processes from
             the saved checkpoint
    best_stable.zip  lowest all9 bunching among checkpoints with 0 safety violations
             (B1 and all9), all9 never_inserted <= C_all's + 1 and all9 person_delay_h
             <= C_all's (ties: lower all9 person_delay_h); see better_stable
G's run folder: runs/ppo_G_w<w>_e<w_early>_c<c_switch>_<obs>_s<seed>_<steps/1000>k.

Agent G2 (--reward regular_g2): reward = --w-delay x E - --w-headway x headway deviation
of each bus passing B1 - --c-switch per switch (reward.py); mask and all9 validation as
G; extra checkpoint best_bunch.zip (see better_bunch): lowest mean of B1 and all9
bunching among checkpoints with 0 violations, all9 never_inserted <= C_all's + 1 and
person delay at B1 and at all 9 <= 1.05 x C_all's.
    python bo4_bus/rl/train_ppo.py --reward regular_g2 --w-headway 4 --c-switch 0.5 \
                                   --obs v2 --train-seed 0
Run folder: runs/ppo_G2_h<w_headway>_d<w_delay>_c<c_switch>_<obs>_s<seed>_<steps/1000>k.

Agent H (--reward regular_h): residual on rule C (h_agent.py, HEnv): at B1, while a bus is
within 150 m, every 5 s choose follow C / serve the bus / hold the bus; G2's reward
(--w-headway, --w-delay, --c-switch), mask, all9 validation and best_bunch as G2.
    python bo4_bus/rl/train_ppo.py --reward regular_h --w-headway 4 --c-switch 0.5 --train-seed 0
Run folder: runs/ppo_H_h<w_headway>_d<w_delay>_c<c_switch>_s<seed>_<steps/1000>k.

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
    best_regular.zip         rewards F and G; best.json then also holds "best_regular"
                             (None until a checkpoint qualifies) and "c_all_reference"
    best_stable.zip          with --val-all9 (G); best.json "best_stable"
monitor.csv also logs per episode: switches, shield overrides (min / max green),
safety violations and teleports at B1 (env.EPISODE_KEYS).
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
from env import EPISODE_KEYS, TRAIN_SEEDS, BusPriorityEnv  # noqa: E402
from c_experiments import _load, _save  # noqa: E402

SELECT = {"vehicle": "veh_delay_B1", "person": "person_delay_B1_h",
          "regular": "person_delay_B1_h", "regular_g": "person_delay_B1_h",
          "regular_g2": "person_delay_B1_h", "regular_h": "person_delay_B1_h"}
LETTER = {"vehicle": "D", "person": "E", "regular": "F", "regular_g": "G", "regular_g2": "G2",
          "regular_h": "H"}
REGULAR_REWARDS = ("regular", "regular_g", "regular_g2", "regular_h")
MASKED_REWARDS = ("regular_g", "regular_g2", "regular_h")   # mask + all9 validation by default
HEADWAY_REWARDS = ("regular_g2", "regular_h")                # G2's reward, best_bunch
VAL_COLS = ["bus_delay_B1", "veh_delay_B1", "person_delay_B1_h", "bunch_last_total",
            "cv_last_mean", "car_delay_total", "person_delay_h", "bus_jdelay", "cross_delay",
            "rl_shield_overrides", "rl_violations", "never_inserted", "teleports"]


def make_env(reward, rank, w=1.0, obs="v1", w_early=0.0, c_switch=0.0, w_delay=1.0,
             w_headway=0.0):
    def _init():
        if reward == "regular_h":
            from h_agent import HEnv
            return HEnv(w_delay=w_delay, w_headway=w_headway, c_switch=c_switch)
        return BusPriorityEnv(reward=reward, w=w, obs=obs, w_early=w_early, c_switch=c_switch,
                              w_delay=w_delay, w_headway=w_headway)
    return _init


def run_name(reward, w, obs, train_seed, steps, w_early=None, c_switch=None, mask=None,
             w_headway=None, w_delay=1.0):
    """runs/ folder name, e.g. ppo_F_w1_v2_s0_200k (w only for F and G),
    ppo_G_w5_e1_c0.5_v2_s0_200k (G; '_nomask' if trained without the mask)."""
    if reward == "regular_g2":
        w_part = (f"_h{w_headway:g}_d{w_delay:g}_c{c_switch:g}" + ("" if mask else "_nomask"))
        return f"ppo_G2{w_part}_{obs}_s{train_seed}_{steps // 1000}k"
    if reward == "regular_h":
        w_part = (f"_h{w_headway:g}_d{w_delay:g}_c{c_switch:g}" + ("" if mask else "_nomask"))
        return f"ppo_H{w_part}_s{train_seed}_{steps // 1000}k"
    w_part = f"_w{w:g}" if reward in REGULAR_REWARDS else ""
    if reward == "regular_g":
        w_part += f"_e{w_early:g}_c{c_switch:g}" + ("" if mask else "_nomask")
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


def _val_job(job):
    """ProcessPool worker: one validation run of a saved checkpoint (or C_all).
    Each (run, checkpoint, scope, attempt) writes its own raw folder; a run whose SUMO
    output cannot be read (e.g. a truncated tripinfo.xml) is retried once in a fresh
    folder instead of aborting the training."""
    for attempt in (1, 2):
        tag = (f"rl_val_{job['run']}_{job['steps']}_{job['scope']}"
               + ("" if attempt == 1 else f"_retry{os.getpid()}"))
        try:
            row, _ = ev.run_one(job["name"], job["seed"], model_path=job["model"],
                                duration=job["duration"], scope=job["scope"], tag=tag)
            if row["backend"] != "libsumo":
                # Bus_Demo.import_traci falls back to traci when libsumo cannot be loaded
                # (seen under memory pressure); such a row is not valid, so retry
                raise RuntimeError(f"backend {row['backend']} instead of libsumo")
            return job, row
        except Exception as e:                      # noqa: BLE001
            if attempt == 2:
                # plain-text error: libsumo exceptions may hold objects that cannot be
                # sent back from the worker process (and would hide the real error)
                import traceback
                raise RuntimeError(f"validation run {job['name']} {job['steps']} "
                                   f"{job['scope']} seed {job['seed']} failed twice:\n"
                                   + traceback.format_exc()) from None
            print(f"[val] {job['name']} {job['steps']} {job['scope']} seed {job['seed']}: "
                  f"{type(e).__name__}: {e}; retrying once", flush=True)


def validate_pool(out, name, steps, model_path, val_duration, scopes=("b1",), workers=4):
    """Like validate(), from a saved checkpoint, in `workers` processes, for each scope.
    validation.csv gets a "scope" column. Returns the B1-scope means (VAL_COLS) plus the
    means of the other scopes prefixed "<scope>_" (e.g. all9_bunch_last_total)."""
    from concurrent.futures import ProcessPoolExecutor
    path = os.path.join(out, "validation.csv")
    rows = _load(path)
    done = {(r["name"], int(r["steps"]), r.get("scope", "b1"), int(r["seed"])) for r in rows}
    jobs = [dict(name=name, seed=s, model=model_path, duration=val_duration, scope=sc,
                 run=os.path.basename(out), steps=steps)
            for sc in scopes for s in ev.VAL_SEEDS if (name, steps, sc, s) not in done]
    if jobs:
        with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
            for job, row in ex.map(_val_job, jobs):
                assert row["backend"] == "libsumo"
                rows.append({"steps": steps, "scope": job["scope"],
                             **{k: row[k] for k in ["name", "seed", "wall_s"] + VAL_COLS}})
                _save(rows, path)
    m = {}
    for sc in scopes:
        cur = pd.DataFrame([r for r in rows if r["name"] == name and int(r["steps"]) == steps
                            and r.get("scope", "b1") == sc])
        prefix = "" if sc == "b1" else f"{sc}_"
        m.update({prefix + k: v for k, v in cur[VAL_COLS].mean().to_dict().items()})
    return m


REGULAR_KEY, REGULAR_GUARD = "bunch_last_total", "person_delay_B1_h"
STABLE_KEY = "all9_bunch_last_total"


def better_stable(m, best, c_ref):
    """Agent G's best_stable checkpoint: is validation result `m` (with all9_* means)
    better than `best`? Qualifies only with 0 safety violations at B1 and at all 9,
    all9 never_inserted <= C_all's + 1 and all9 person_delay_h <= C_all's (C_all = C at
    all 9, the same runs as c_ref); then lowest all9 bunching, ties: all9 person delay."""
    ok = (m["rl_violations"] == 0 and m["all9_rl_violations"] == 0
          and m["all9_never_inserted"] <= c_ref["never_inserted"] + 1
          and m["all9_person_delay_h"] <= c_ref["person_delay_h"])
    if not ok:
        return False
    return best is None or ((m[STABLE_KEY], m["all9_person_delay_h"])
                            < (best[STABLE_KEY], best["all9_person_delay_h"]))


def better_regular(m, best, c_ref):
    """Reward F's second checkpoint: is validation result `m` better than `best` (None
    if no checkpoint qualified yet)? Only checkpoints whose person delay at B1 is not
    worse than C_all's qualify; then lowest bunching, ties broken by person delay."""
    if m[REGULAR_GUARD] > c_ref[REGULAR_GUARD]:
        return False
    return best is None or ((m[REGULAR_KEY], m[REGULAR_GUARD])
                            < (best[REGULAR_KEY], best[REGULAR_GUARD]))


BUNCH_DELAY_TOL = 1.05      # best_bunch: person delay may be up to 5 % above C_all's


def bunch_score(m):
    """Mean of last-stop bunching at B1 and at all 9 junctions (validation means)."""
    return (m["bunch_last_total"] + m["all9_bunch_last_total"]) / 2


def better_bunch(m, best, c_ref):
    """Agent G2's best_bunch checkpoint (bunching first): qualifies with 0 safety
    violations (B1, all9), all9 never_inserted <= C_all's + 1 and person delay at B1 and
    at all 9 <= BUNCH_DELAY_TOL x C_all's; then lowest bunch_score, ties: lower all9
    person delay."""
    ok = (m["rl_violations"] == 0 and m["all9_rl_violations"] == 0
          and m["all9_never_inserted"] <= c_ref["never_inserted"] + 1
          and m["person_delay_B1_h"] <= BUNCH_DELAY_TOL * c_ref["person_delay_B1_h"]
          and m["all9_person_delay_h"] <= BUNCH_DELAY_TOL * c_ref["person_delay_h"])
    if not ok:
        return False
    return best is None or ((bunch_score(m), m["all9_person_delay_h"])
                            < (bunch_score(best), best["all9_person_delay_h"]))


def make_callback(out, reward, every, val_duration, obs="v1", c_ref=None, val_all9=False,
                  val_workers=None):
    """c_ref: C_all's validation means (reference_c), needed for rewards F and G.
    val_workers: None = validate in this process (as for D, E, F), else validate the
    saved checkpoint in that many processes (and at all 9 junctions if val_all9)."""
    from stable_baselines3.common.callbacks import BaseCallback
    regular = reward in REGULAR_REWARDS
    bunch = reward in HEADWAY_REWARDS and val_all9
    assert not regular or c_ref is not None
    assert not val_all9 or (c_ref is not None and val_workers)

    class ValidationCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = 0
            self.best = None
            self.best_regular = None
            self.best_stable = None
            self.best_bunch = None
            self.resumed = False          # set by --resume: start checkpoint already validated
            self.summary_path = os.path.join(out, "validation_summary.csv")
            self.summary = _load(self.summary_path)
            self._restore()

        def _restore(self):
            """Best-so-far from validation_summary.csv (same rules as _evaluate), so a
            checkpoint validated later (--validate-ckpt) competes with the earlier ones."""
            key = SELECT[reward]
            for r in self.summary:
                m = {k: v for k, v in r.items()
                     if k not in ("best", "best_regular", "best_stable", "best_bunch",
                                  "val_wall_s")}
                if self.best is None or m[key] < self.best[key]:
                    self.best = {"steps": int(m["steps"]), key: m[key], **m}
                if regular and better_regular(m, self.best_regular, c_ref):
                    self.best_regular = {"steps": int(m["steps"]), REGULAR_KEY: m[REGULAR_KEY], **m}
                if val_all9 and better_stable(m, self.best_stable, c_ref):
                    self.best_stable = {"steps": int(m["steps"]), STABLE_KEY: m[STABLE_KEY], **m}
                if bunch and better_bunch(m, self.best_bunch, c_ref):
                    self.best_bunch = {"steps": int(m["steps"]), "bunch_score": bunch_score(m), **m}

        def _evaluate(self):
            steps = int(self.num_timesteps)
            model = self.model
            tic = time.time()
            path = os.path.join(out, f"ckpt_{steps}.zip")
            if val_workers:
                model.save(path)
                m = validate_pool(out, "RL", steps, path, val_duration,
                                  ("b1", "all9") if val_all9 else ("b1",), val_workers)
            else:
                m = validate(out, "RL", steps, ev.as_policy(model), val_duration, obs)
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
            is_best_stable = val_all9 and better_stable(m, self.best_stable, c_ref)
            if is_best_stable:
                self.best_stable = {"steps": steps, STABLE_KEY: m[STABLE_KEY], **m}
                model.save(os.path.join(out, "best_stable.zip"))
            is_best_bunch = bunch and better_bunch(m, self.best_bunch, c_ref)
            if is_best_bunch:
                self.best_bunch = {"steps": steps, "bunch_score": bunch_score(m), **m}
                model.save(os.path.join(out, "best_bunch.zip"))
            if is_best or is_best_regular or is_best_stable or is_best_bunch:
                self._write_best()
            for k, v in m.items():
                self.logger.record(f"val/{k}", v)
            self.logger.record("val/is_best", float(is_best))
            extra = {}
            if regular:
                self.logger.record("val/is_best_regular", float(is_best_regular))
                extra = {"best_regular": is_best_regular}
            if val_all9:
                self.logger.record("val/is_best_stable", float(is_best_stable))
                extra["best_stable"] = is_best_stable
            if bunch:
                self.logger.record("val/is_best_bunch", float(is_best_bunch))
                extra["best_bunch"] = is_best_bunch
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
            if val_all9:
                bs = self.best_stable
                msg += (f"; all9: bunching {m[STABLE_KEY]:.1f}, person delay "
                        f"{m['all9_person_delay_h']:.1f} h, never inserted "
                        f"{m['all9_never_inserted']:.1f}, violations {m['all9_rl_violations']:.0f}, "
                        f"overrides B1 {m['rl_shield_overrides']:.0f} / all9 "
                        f"{m['all9_rl_shield_overrides']:.0f}; best_stable "
                        + (f"@ {bs['steps']}" if bs else "none yet"))
            if bunch:
                bb = self.best_bunch
                msg += (f"; bunch score {bunch_score(m):.1f} (C_all {c_ref['bunch_last_total']:.1f})"
                        f", best_bunch " + (f"@ {bb['steps']}" if bb else "none yet"))
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
            if val_all9:
                best.update(best_stable=self.best_stable,
                            best_stable_rule="lowest all9 bunch_last_total (ties: all9 "
                                             "person_delay_h) among checkpoints with 0 violations "
                                             "(B1, all9), all9 never_inserted <= C_all's + 1, "
                                             "all9 person_delay_h <= C_all's")
            if bunch:
                best.update(best_bunch=self.best_bunch,
                            best_bunch_rule="lowest mean of B1 and all9 bunch_last_total (ties: "
                                            "all9 person_delay_h) among checkpoints with 0 "
                                            "violations, all9 never_inserted <= C_all's + 1, "
                                            f"person delay at B1 and all9 <= {BUNCH_DELAY_TOL} x "
                                            "C_all's")
            with open(os.path.join(out, "best.json"), "w") as f:
                json.dump(best, f, indent=1)

        def _on_training_start(self):
            if not self.resumed:
                self._evaluate()                              # untrained policy

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


def reference_c(out, val_duration, workers=None):
    """C at B1 (= C at all 9 junctions) on the validation seeds (once per run folder)."""
    if workers:
        m = validate_pool(out, "C_all", 0, None, val_duration, ("b1",), workers)
    else:
        m = validate(out, "C_all", 0, None, val_duration)
    print("[val] C at B1 reference: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
    return m


def latest_validated_ckpt(out):
    """Steps of the newest ckpt_<steps>.zip that has a row in validation_summary.csv."""
    done = {int(r["steps"]) for r in _load(os.path.join(out, "validation_summary.csv"))}
    have = [s for s in done if os.path.exists(os.path.join(out, f"ckpt_{s}.zip"))]
    return max(have) if have else None


def check_config(out, a):
    """The options that define a run must equal its config.json (resume / validate)."""
    with open(os.path.join(out, "config.json")) as f:
        cfg = json.load(f)
    for k in ("reward", "w", "w_early", "c_switch", "w_headway", "w_delay", "obs", "mask",
              "val_all9", "val_duration", "train_seed", "steps", "n_envs", "n_steps",
              "eval_every"):
        if k in cfg and cfg.get(k) != getattr(a, k):
            raise SystemExit(f"--{k.replace('_', '-')} {getattr(a, k)} differs from the run's "
                             f"config.json ({cfg.get(k)}); use the training command's options")


def validate_checkpoint(out, a, val_workers):
    """Finish an interrupted run: validate ckpt_<steps>.zip like the training callback
    would have (same rules, best-so-far restored from validation_summary.csv). Rows
    already in validation.csv are not rerun."""
    ckpt = os.path.join(out, f"ckpt_{a.validate_ckpt}.zip")
    if not os.path.exists(ckpt):
        raise SystemExit(f"{ckpt} not found")
    with open(os.path.join(out, "config.json")) as f:
        cfg = json.load(f)
    for k in ("reward", "w", "w_early", "c_switch", "w_headway", "w_delay", "obs", "mask",
              "val_all9", "val_duration"):
        if cfg.get(k) != getattr(a, k):
            raise SystemExit(f"--{k.replace('_', '-')} {getattr(a, k)} differs from the run's "
                             f"config.json ({cfg.get(k)}); use the training command's options")
    from stable_baselines3.common.logger import configure
    if a.mask:
        from sb3_contrib import MaskablePPO as Algo
    else:
        from stable_baselines3 import PPO as Algo
    model = Algo.load(ckpt, device="cpu")
    model.set_logger(configure(os.path.join(out, "tb", "validate_ckpt"), ["csv"]))
    c_ref = reference_c(out, a.val_duration, val_workers)       # from validation.csv
    cb = make_callback(out, a.reward, a.eval_every, a.val_duration, a.obs, c_ref,
                       a.val_all9, val_workers)
    cb.model = model
    cb.num_timesteps = a.validate_ckpt
    cb._evaluate()
    print(f"validated {ckpt}; best checkpoints in {out}/best.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="vehicle",
                    choices=["vehicle", "person", "regular", "regular_g", "regular_g2",
                             "regular_h"],
                    help="vehicle = D, person = E, regular = F, regular_g = G, regular_g2 = G2")
    ap.add_argument("--w", type=float, default=1.0, help="lateness weight of rewards F and G")
    ap.add_argument("--w-early", type=float, default=None, help="G: early-bus penalty weight")
    ap.add_argument("--c-switch", type=float, default=None, help="G, G2: switch cost weight")
    ap.add_argument("--w-headway", type=float, default=None, help="G2: headway penalty weight")
    ap.add_argument("--w-delay", type=float, default=1.0, help="G2: weight of reward E")
    ap.add_argument("--mask", action=argparse.BooleanOptionalAction, default=None,
                    help="MaskablePPO with the shield's action mask (default: on for G only)")
    ap.add_argument("--val-all9", action=argparse.BooleanOptionalAction, default=None,
                    help="also validate checkpoints at all 9 junctions (default: on for G only)")
    ap.add_argument("--val-workers", type=int, default=4,
                    help="processes for validation with --val-all9 / --mask runs")
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
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run of the same options from its latest "
                         "validated ckpt_<steps>.zip up to --steps")
    ap.add_argument("--validate-ckpt", type=int, default=None, metavar="STEPS",
                    help="no training: validate runs/<name>/ckpt_STEPS.zip of an existing run "
                         "(same options as its training) and update its best_* checkpoints")
    a = ap.parse_args()
    g = a.reward in MASKED_REWARDS
    g2 = a.reward in HEADWAY_REWARDS
    if a.reward not in ("regular", "regular_g") and a.w != 1.0:
        ap.error("--w only applies to --reward regular / regular_g")
    if a.reward == "regular_g" and (a.w_early is None or a.c_switch is None):
        ap.error("--reward regular_g needs --w-early and --c-switch")
    if g2 and (a.w_headway is None or a.c_switch is None):
        ap.error("--reward regular_g2 / regular_h needs --w-headway and --c-switch")
    if a.reward != "regular_g" and a.w_early is not None:
        ap.error("--w-early only applies to --reward regular_g")
    if not g and a.c_switch is not None:
        ap.error("--c-switch only applies to --reward regular_g / regular_g2")
    if not g2 and (a.w_headway is not None or a.w_delay != 1.0):
        ap.error("--w-headway / --w-delay only apply to --reward regular_g2 / regular_h")
    a.mask = g if a.mask is None else a.mask
    a.val_all9 = g if a.val_all9 is None else a.val_all9
    if a.val_all9 and a.reward not in REGULAR_REWARDS:
        ap.error("--val-all9 needs reward F or G (best_stable uses the C_all reference)")
    val_workers = a.val_workers if (a.val_all9 or a.mask) else None

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    torch.set_num_threads(2)

    name = a.name or run_name(a.reward, a.w, a.obs, a.train_seed, a.steps, a.w_early,
                              a.c_switch, a.mask, a.w_headway, a.w_delay)
    out = os.path.join(HERE, "runs", name)
    if a.validate_ckpt:
        return validate_checkpoint(out, a, val_workers)
    resume_from = latest_validated_ckpt(out) if a.resume else None
    if a.resume and resume_from is None:
        raise SystemExit(f"--resume: no validated checkpoint in {out}")
    if a.resume:
        check_config(out, a)
        if resume_from >= a.steps:
            raise SystemExit(f"--resume: {out} already reached {resume_from} steps (--steps {a.steps})")
    elif os.path.exists(os.path.join(out, "validation_summary.csv")):
        raise SystemExit(f"{out} already holds a trained run; training again would overwrite "
                         f"its checkpoints while reusing its old validation rows. Use another "
                         f"--name / --train-seed, or --validate-ckpt to finish that run.")
    os.makedirs(out, exist_ok=True)
    cfg = {**vars(a), "name": name, "agent": LETTER[a.reward],
           "w": a.w if a.reward in REGULAR_REWARDS else None,
           "algo": "MaskablePPO" if a.mask else "PPO",
           "obs_size": 43 if a.reward == "regular_h" else int(ob.bounds(a.obs)[0].size),
           "select": SELECT[a.reward], "train_sumo_seeds": list(TRAIN_SEEDS),
           "val_seeds": ev.VAL_SEEDS}
    if a.resume:
        with open(os.path.join(out, "config.json")) as f:
            cfg = json.load(f)                        # keep the original options
        cfg["resumed_from"] = cfg.get("resumed_from", []) + [resume_from]
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    c_ref = reference_c(out, a.val_duration, val_workers)
    vec = VecMonitor(SubprocVecEnv([make_env(a.reward, i, a.w, a.obs, a.w_early or 0.0,
                                             a.c_switch or 0.0, a.w_delay, a.w_headway or 0.0)
                                    for i in range(a.n_envs)],
                                   start_method="spawn"),
                     os.path.join(out, "monitor.csv"), info_keywords=EPISODE_KEYS)
    if a.mask:
        from sb3_contrib import MaskablePPO as Algo
    else:
        Algo = PPO
    cb = make_callback(out, a.reward, a.eval_every, a.val_duration, a.obs, c_ref,
                       a.val_all9, val_workers)
    if a.resume:
        # weights + optimizer state from the checkpoint; the envs restart on new episodes
        # (env RNGs re-seeded from train_seed + resume step), so the continuation is
        # reproducible but not identical to an uninterrupted run
        model = Algo.load(os.path.join(out, f"ckpt_{resume_from}.zip"), env=vec, device="cpu",
                          tensorboard_log=os.path.join(out, "tb"))
        model.set_random_seed(a.train_seed + resume_from)
        cb.next_eval = resume_from + a.eval_every
        cb.resumed = True
        print(f"resuming {out} from ckpt_{resume_from}.zip", flush=True)
    else:
        model = Algo("MlpPolicy", vec, n_steps=a.n_steps, batch_size=64, n_epochs=10,
                     learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                     seed=a.train_seed, verbose=0, device="cpu",
                     tensorboard_log=os.path.join(out, "tb"))
    tic = time.time()
    model.learn(total_timesteps=a.steps - (resume_from or 0), callback=cb, tb_log_name="ppo",
                reset_num_timesteps=not a.resume)
    model.save(os.path.join(out, "final_model.zip"))
    vec.close()
    print(f"trained {model.num_timesteps} steps in {(time.time() - tic) / 60:.1f} min; "
          f"best checkpoint in {out}/best.json", flush=True)


if __name__ == "__main__":
    main()
