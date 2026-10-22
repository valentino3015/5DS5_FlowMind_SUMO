"""
Evaluate a B1 policy with the step-4 measures: the policy drives B1 inside the existing
grid runner (Grid_Demo.run_seed, unchanged), controller C (C_tuned) runs the other 8
junctions, so every network metric is computed exactly as in results/c_final_*.csv.

    python bo4_bus/rl/evaluate.py final --model RL_D=runs/ppo_D_200k/best_model.zip \
                                        --model RL_E=runs/ppo_E_200k/best_model.zip [--workers 4]
    python bo4_bus/rl/evaluate.py val --model RL_F_w1=runs/ppo_F_w1_v2_s0_200k/best_model.zip ...
    python bo4_bus/rl/evaluate.py final                   # only the all-C reference
(--model-d PATH / --model-e PATH are kept as aliases of --model RL_D=PATH / RL_E=PATH.)

Every --model LABEL=PATH is a PPO policy at B1 + C elsewhere; C_all (C at all 9
junctions, = C_tuned of step 4) is always run as the reference. The observation
version (v1 / v2) of each model is read from its input size.
    final  evaluation seeds 10000-10019, warm-up 600 s + 7200 s
           -> results/rl_final_runs.csv, rl_final_lines.csv, rl_final_summary.csv
    val    validation seeds 900-904 (model selection, e.g. the weight w of reward F),
           warm-up 600 s + 7200 s -> results/rl_val_runs.csv, rl_val_lines.csv,
           rl_val_summary.csv, plus a ranking of the models (see rank())
One row per finished run; a restart skips finished runs. A label identifies a model in
these CSVs, so give a different model a new label.

B1-only measures added to every row (from the run's raw outputs, before deletion):
    bus_delay_B1     mean delay of buses on the stop-to-stop segment that crosses B1
                     (the team's "bus delay per junction", Grid_Demo seg_delay), s
    veh_delay_B1     (car time loss on B1's 4 approaches + bus delay at B1)
                     / (cars that left those approaches + bus passages), s per vehicle
    person_delay_B1_h  (1.3 x car time loss on B1's approaches
                       + sum of occupancy x bus delay at B1) / 3600, person-hours
    rl_shield_overrides, rl_violations   B1 shield overrides and safety violations
"""
import argparse
import contextlib
import math
import os
import shutil
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env import TLS, B1Signal  # noqa: E402  (also puts bo4_bus/ on sys.path)
import Grid_Demo as gd  # noqa: E402
import bus_metrics as bm  # noqa: E402
import c_experiments as ce  # noqa: E402
import gen_grid_routes as gen  # noqa: E402
import rule_tsp  # noqa: E402
from build_grid import LINES  # noqa: E402
import observation as ob  # noqa: E402
from sense import Geometry, Sensor  # noqa: E402

RESULTS = gd.RESULTS
VAL_SEEDS = list(range(900, 905))
EVAL_SEEDS = ce.EVAL_SEEDS
B1_COLS = ["bus_delay_B1", "veh_delay_B1", "person_delay_B1_h"]


class RLController:
    """Grid_Demo controller for B1: fixed-time program during warm-up, then the policy
    every `delta` s through B1Signal (same shield / transitions / monitor as the env)."""

    def __init__(self, policy, net, warmup=600.0, delta=5, obs_version="v1"):
        self.policy, self.warmup, self.delta = policy, warmup, delta
        self.obs_version = obs_version
        self.sensor = Sensor(Geometry(net))
        self.signal = B1Signal(TLS, delta)
        self.log, self.outcomes = [], []      # read by Grid_Demo for TSP controllers
        self.next_tick = None

    def reset(self, sim):
        pass

    def act(self, t, sim):
        if not self.signal.active:
            if t < self.warmup or not self.signal.ready(sim):
                return
            self.signal.take_over(sim, t)
            self.sensor.reset()
            self.next_tick = t
        else:
            self.signal.on_second(sim, t)
        if t >= self.next_tick:
            s = self.sensor.sense(sim, t, self.signal.current_green,
                                  self.signal.time_in_phase(t), False)
            self.signal.apply(sim, t, self.policy(ob.observe(s, version=self.obs_version)))
            self.next_tick += self.delta


@contextlib.contextmanager
def b1_policy(policy, holder, obs_version="v1"):
    """Within the block, rule_tsp.make_controllers puts RLController(policy) at B1."""
    orig = rule_tsp.make_controllers

    def make(mode, net, params, fixed_cls):
        ctrls = orig(mode, net, params, fixed_cls)
        ctrls[TLS] = holder["ctrl"] = RLController(policy, net, obs_version=obs_version)
        return ctrls

    rule_tsp.make_controllers = make
    try:
        yield
    finally:
        rule_tsp.make_controllers = orig


def sb3_policy(model_path):
    """(policy obs -> action, observation version read from the model's input size)."""
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    version = ob.version_of_size(model.observation_space.shape[0])
    return (lambda obs: int(model.predict(obs, deterministic=True)[0])), version


def b1_measures(raw, routes, geo, t0, t1):
    """B1-only delays from a run's raw outputs (see module docstring)."""
    stops = bm.parse_stopinfo(os.path.join(raw, "stopinfo.xml"))
    occ = {v: p["occupancy"] for v, p in bm.parse_bus_params(routes).items()}
    bus = []                                             # (delay, occupancy) per bus passage
    for d, spec in LINES.items():
        k = spec["junctions"].index(TLS)
        seg = bm.segment_delays(stops, spec["stops"], geo["ff"][d], t0, t1)[k]
        bus += [(x, occ.get(v, 0.0)) for v, x in seg.items()]
    ed = bm.parse_edgedata(os.path.join(raw, "edgedata.xml"))
    appr = [edges for a, (tls, edges) in geo["approaches"].items() if tls == TLS]
    car_loss = sum(ed.get(e, {}).get("timeLoss", 0.0) for edges in appr for e in edges)
    car_n = sum(ed.get(edges[-1], {}).get("left", 0) for edges in appr)
    n = car_n + len(bus)
    return {"bus_delay_B1": bm.mean_or_nan([x for x, _ in bus]),
            "veh_delay_B1": (car_loss + sum(x for x, _ in bus)) / n if n else math.nan,
            "person_delay_B1_h": (gen.CAR_OCCUPANCY * car_loss + sum(x * o for x, o in bus)) / 3600.0,
            "bus_passages_B1": len(bus), "car_passages_B1": car_n}


_GEO, _POLICIES = None, {}


def run_one(name, seed, policy=None, model_path=None, duration=7200.0, tag=None,
            obs_version="v1"):
    """One grid run with `policy` (callable obs -> action, fed `obs_version`
    observations) or `model_path` (version read from the model) at B1; neither = C at
    B1. Returns (row, line_rows) like c_experiments.run_job."""
    global _GEO
    if _GEO is None:
        _GEO = gd.build_geo()
    if policy is None and model_path:
        if model_path not in _POLICIES:
            _POLICIES[model_path] = sb3_policy(model_path)
        policy, obs_version = _POLICIES[model_path]
    p = ce.C_TUNED
    tag = tag or f"rl_{name}"
    args = gd.parse_args(["--controller", "tsp_cond", "--seeds", str(seed), "--tag", tag,
                          "--late-thr", str(p["late_thr"]), "--queue-thr", str(p["queue_thr"]),
                          "--max-ext", str(p["max_ext"]), "--duration", str(duration)])
    holder = {}
    tic = time.time()
    with (b1_policy(policy, holder, obs_version) if policy else contextlib.nullcontext()):
        row, lines, _, _ = gd.run_seed(seed, args, _GEO)
    row["wall_s"] = round(time.time() - tic, 1)
    raw = os.path.join(RESULTS, "raw", tag, f"seed_{seed}")
    route = os.path.join(gd.HERE, "routes", tag, f"routes_seed{seed}.rou.xml")
    row.update(b1_measures(raw, route, _GEO, args.warmup, args.warmup + args.duration))
    ctrl = holder.get("ctrl")
    row["rl_shield_overrides"] = len(ctrl.signal.shield_log) if ctrl else math.nan
    row["rl_violations"] = len(ctrl.signal.violations) if ctrl else math.nan
    row["obs_version"] = ctrl.obs_version if ctrl else ""
    shutil.rmtree(raw, ignore_errors=True)
    if os.path.exists(route):
        os.remove(route)
    ident = {"name": name, "flow": args.entry_flow, "late_thr": p["late_thr"],
             "queue_thr": p["queue_thr"], "max_ext": p["max_ext"]}
    row = {**ident, **row}
    row["bunch_last_total"] = sum(row[f"bunch_last_{d}"] for d in LINES)
    row["cv_last_mean"] = sum(row[f"cv_last_{d}"] for d in LINES) / len(LINES)
    row["bus_tt"] = sum(row[f"tt_{d}"] for d in LINES) / len(LINES)
    return row, [{**ident, **r} for r in lines]


def final_job(job):
    """ProcessPool worker for c_experiments.run_all: job carries name, seed, model, exp."""
    row, lines = run_one(job["name"], job["seed"], model_path=job.get("model"),
                         tag=f"{job['exp']}_{job['name']}")
    return row, lines, []


SEEDS = {"final": EVAL_SEEDS, "val": VAL_SEEDS}


def evaluate(what, models, workers):
    """models: {label: model path}, run after C_all in the given order."""
    names = ["C_all"] + list(models)
    seeds = SEEDS[what]
    jobs = [dict(name=n, seed=s, flow=ce.FLOW, model=models.get(n), controller="rl")
            for n in names for s in seeds]
    ce.N_WORKERS = workers
    exp = f"rl_{what}"
    df, _, _ = ce.run_all(jobs, exp, os.path.join(RESULTS, f"{exp}_runs.csv"),
                          os.path.join(RESULTS, f"{exp}_lines.csv"), fn=final_job)
    assert (df.backend == "libsumo").all()
    title = "Final evaluation" if what == "final" else "Validation"
    s = summarize(df, names, os.path.join(RESULTS, f"{exp}_summary.csv"),
                  f"{title}, seeds {seeds[0]}-{seeds[-1]}")
    if what == "val":
        rank(s)


def rank(s, key="bunch_last_total", guard="person_delay_B1_h"):
    """Model selection on the validation seeds: lowest mean `key` (bunching, the BO4
    objective) among the models whose mean `guard` (person delay at B1) is not worse
    than C_all's; ties broken by `guard`. Prints the ranking, returns the picked label."""
    ref = s.loc[s.name == "C_all", guard].iloc[0]
    m = s[s.name != "C_all"].copy()
    if m.empty:
        return None
    m["ok"] = m[guard] <= ref
    m = m.sort_values(["ok", key, guard], ascending=[False, True, True])
    print(f"\nRanking (lowest {key}, subject to {guard} <= C_all's {ref:.3f}):")
    for _, r in m.iterrows():
        note = "ok" if r["ok"] else f"worse than C_all on {guard}"
        print(f"  {r['name']:<20} {key} {r[key]:7.2f} +-{r[key + '_ci95']:5.2f}   "
              f"{guard} {r[guard]:7.3f}   {note}")
    best = m.iloc[0]
    print(f"Pick: {best['name']}" + ("" if best["ok"] else "  (no model passes the guard)"))
    return best["name"]


def summarize(df, names, path, title):
    cols = ce.SUMMARY_COLS + B1_COLS + ["rl_shield_overrides", "rl_violations"]
    base = df[df.name == "C_all"].set_index("seed")
    out = []
    for n in names:
        sub = df[df.name == n].set_index("seed")
        r = {"name": n, "n_seeds": len(sub)}
        for c in cols:
            if c in sub:
                r[c], r[f"{c}_ci95"] = ce.ci95(sub[c])
                if n != "C_all" and c in base:
                    diff = (sub[c] - base[c]).dropna()
                    r[f"{c}_diff"], r[f"{c}_diff_ci95"] = ce.ci95(diff)
        out.append(r)
    s = pd.DataFrame(out)
    s.to_csv(path, index=False)
    print(f"\n{title} (mean +- 95 % CI):")
    print(f"{'metric':<20}" + "".join(f"{n:>20}" for n in names))
    for c in cols:
        if c in s:
            print(f"{c:<20}" + "".join(
                f"{r[c]:>12.2f} +-{r[c + '_ci95']:>5.2f}" if not pd.isna(r[c]) else f"{'-':>20}"
                for _, r in s.iterrows()))
    print("\nPaired difference vs C_all:")
    for c in ["bunch_last_total", "car_delay_total", "cross_delay", "person_delay_h"] + B1_COLS:
        print(f"{c:<20}" + "".join(
            f"{r[c + '_diff']:>12.2f} +-{r[c + '_diff_ci95']:>5.2f}" for _, r in s.iterrows()
            if r["name"] != "C_all" and c + "_diff" in r))
    return s


def main():
    ap = argparse.ArgumentParser(description="Evaluate B1 policies with the step-4 measures")
    ap.add_argument("what", choices=["final", "val"],
                    help="final: seeds 10000-10019; val: validation seeds 900-904")
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=PATH",
                    help="PPO model at B1 under LABEL (repeatable), e.g. RL_F=runs/x/best_model.zip")
    ap.add_argument("--model-d", default=None, help="same as --model RL_D=PATH")
    ap.add_argument("--model-e", default=None, help="same as --model RL_E=PATH")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    specs = a.model + [f"{k}={v}" for k, v in (("RL_D", a.model_d), ("RL_E", a.model_e)) if v]
    try:
        models = parse_models(specs)
    except ValueError as e:
        ap.error(str(e))
    evaluate(a.what, models, a.workers)


def parse_models(specs):
    """['RL_F=path', ...] -> {'RL_F': absolute path, ...} in the given order."""
    models = {}
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not (sep and label and path) or label == "C_all" or label in models:
            raise ValueError(f"--model {spec!r}: expected LABEL=PATH with a unique label "
                             f"other than C_all")
        if not os.path.isfile(path):
            raise ValueError(f"--model {spec!r}: no such file")
        models[label] = os.path.abspath(path)
    return models


if __name__ == "__main__":
    main()
