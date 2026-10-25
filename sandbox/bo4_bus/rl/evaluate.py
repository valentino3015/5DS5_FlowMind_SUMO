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

Scope (--scope, default b1):
    b1    the policy at B1 only, C (C_tuned) at the other 8 junctions (as above)
    all9  one copy of the same policy at each of the 9 junctions, decentralised: every
          junction has its own sensor (approaches read from the network, its own
          last-passage memory per line), its own B1Signal shield / transitions / safety
          monitor, and decides every 5 s. No retraining. Model rows are named
          "<label>@all9"; --with-b1 adds the same model at B1 only as "<label>@B1".
          Outputs: final -> results/rl_all9_runs.csv, rl_all9_lines.csv, rl_all9_summary.csv;
          val -> results/rl_all9_val_*.csv.
    python bo4_bus/rl/evaluate.py val --scope all9 --with-b1 --model F_s0=runs/ppo_F_w5_v2_s0_200k/best_regular.zip
    python bo4_bus/rl/evaluate.py smoke --scope all9 --model F_s0=... [--seed 900] [--duration 1800]
          (one run per model, prints safety / teleports / switches per junction, no CSV)

B1-only measures added to every row (from the run's raw outputs, before deletion):
    bus_delay_B1     mean delay of buses on the stop-to-stop segment that crosses B1
                     (the team's "bus delay per junction", Grid_Demo seg_delay), s
    veh_delay_B1     (car time loss on B1's 4 approaches + bus delay at B1)
                     / (cars that left those approaches + bus passages), s per vehicle
    person_delay_B1_h  (1.3 x car time loss on B1's approaches
                       + sum of occupancy x bus delay at B1) / 3600, person-hours
    rl_shield_overrides, rl_violations   shield overrides and safety violations, summed
                     over the RL-controlled junctions
The same measures for every junction J (bus_delay_J only where a line crosses J):
    bus_delay_J, car_delay_J (car time loss on J's approaches / cars that left them,
    s per car), veh_delay_J, person_delay_J_h, and per junction type T in JUNCTION_TYPES
    (B1, A1C1 = L1 only, B0B2 = L2 only, corners = no bus): bus_delay_T, car_delay_T,
    person_delay_T_h (sum over T's junctions). RL junctions also get rl_switches_J,
    rl_shield_J, rl_violations_J.
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
from build_grid import TLS_IDS  # noqa: E402
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
JUNCTION_TYPES = {"B1": ("B1",), "A1C1": ("A1", "C1"), "B0B2": ("B0", "B2"),
                  "corners": ("A0", "A2", "C0", "C2")}
TYPE_COLS = [f"{m}_{t}" for t in JUNCTION_TYPES for m in ("bus_delay", "car_delay")] + \
            [f"person_delay_{t}_h" for t in JUNCTION_TYPES]
SCOPES = ("b1", "all9")


class RLController:
    """Grid_Demo controller for one junction (default B1): fixed-time program during
    warm-up, then the policy every `delta` s through B1Signal (same shield / transitions
    / monitor as the env). Sensor, line memory and shield are per instance, so several
    controllers sharing one policy are fully decentralised."""

    def __init__(self, policy, net, warmup=600.0, delta=5, obs_version="v1", tls=TLS):
        self.policy, self.warmup, self.delta = policy, warmup, delta
        self.obs_version = obs_version
        self.tls = tls
        self.geometry = Geometry(net, tls)
        self.sensor = Sensor(self.geometry)
        self.signal = B1Signal(tls, delta)
        self.switches = 0
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
            obs = ob.observe_at(s, self.geometry, version=self.obs_version)
            action = self.policy(obs, self.signal.mask(t))   # plain PPO ignores the mask
            self.switches += self.signal.apply(sim, t, action)[3]
            self.next_tick += self.delta


@contextlib.contextmanager
def rl_policy(policy, holder, obs_version="v1", scope="b1"):
    """Within the block, rule_tsp.make_controllers puts RLController(policy) at B1
    (scope b1) or a separate RLController(policy) at each of the 9 junctions (all9).
    holder["ctrls"] = {tls: RLController}."""
    orig = rule_tsp.make_controllers
    junctions = (TLS,) if scope == "b1" else tuple(TLS_IDS)

    def make(mode, net, params, fixed_cls):
        ctrls = orig(mode, net, params, fixed_cls)
        holder["ctrls"] = {}
        for tls in junctions:
            ctrls[tls] = holder["ctrls"][tls] = RLController(policy, net, obs_version=obs_version,
                                                             tls=tls)
        return ctrls

    rule_tsp.make_controllers = make
    try:
        yield
    finally:
        rule_tsp.make_controllers = orig


def is_maskable(model_path):
    """True for a sb3-contrib MaskablePPO zip (its policy class module says so)."""
    import json
    import zipfile
    with zipfile.ZipFile(model_path) as z:
        data = json.loads(z.read("data"))
    return "maskable" in str(data.get("policy_class", {}).get("__module__", ""))


def as_policy(model):
    """policy(obs, mask=None) -> action of a loaded SB3 model, deterministic. A
    MaskablePPO model only picks actions allowed by the mask; plain PPO ignores it."""
    if type(model).__name__ == "MaskablePPO":
        return lambda obs, mask=None: int(model.predict(obs, deterministic=True,
                                                        action_masks=mask)[0])
    return lambda obs, mask=None: int(model.predict(obs, deterministic=True)[0])


def sb3_policy(model_path):
    """(policy(obs, mask) -> action, observation version read from the model's input size).
    PPO (agents D, E, F) or MaskablePPO (agent G), detected from the zip."""
    if is_maskable(model_path):
        from sb3_contrib import MaskablePPO as Algo
    else:
        from stable_baselines3 import PPO as Algo
    model = Algo.load(model_path, device="cpu")
    version = ob.version_of_size(model.observation_space.shape[0])
    return as_policy(model), version


def _junction_raw(stops, occ, ed, geo, t0, t1, tls):
    """(bus [(delay, occupancy) per passage], car time loss, cars that left) at `tls`."""
    bus = []                                             # (delay, occupancy) per bus passage
    for d, spec in LINES.items():
        if tls in spec["junctions"]:
            k = spec["junctions"].index(tls)
            seg = bm.segment_delays(stops, spec["stops"], geo["ff"][d], t0, t1)[k]
            bus += [(x, occ.get(v, 0.0)) for v, x in seg.items()]
    appr = [edges for a, (j, edges) in geo["approaches"].items() if j == tls]
    car_loss = sum(ed.get(e, {}).get("timeLoss", 0.0) for edges in appr for e in edges)
    car_n = sum(ed.get(edges[-1], {}).get("left", 0) for edges in appr)
    return bus, car_loss, car_n


def _delays(bus, car_loss, car_n):
    n = car_n + len(bus)
    return {"bus_delay": bm.mean_or_nan([x for x, _ in bus]),
            "car_delay": car_loss / car_n if car_n else math.nan,
            "veh_delay": (car_loss + sum(x for x, _ in bus)) / n if n else math.nan,
            "person_delay_h": (gen.CAR_OCCUPANCY * car_loss + sum(x * o for x, o in bus)) / 3600.0,
            "bus_passages": len(bus), "car_passages": car_n}


def junction_measures(raw, routes, geo, t0, t1):
    """Per-junction and per-junction-type delays from a run's raw outputs (see module
    docstring). bus_delay_B1 / veh_delay_B1 / person_delay_B1_h are the columns the
    b1 scope always had."""
    stops = bm.parse_stopinfo(os.path.join(raw, "stopinfo.xml"))
    occ = {v: p["occupancy"] for v, p in bm.parse_bus_params(routes).items()}
    ed = bm.parse_edgedata(os.path.join(raw, "edgedata.xml"))
    per = {j: _junction_raw(stops, occ, ed, geo, t0, t1, j) for j in TLS_IDS}
    row = {}
    for j, raw_j in per.items():
        for k, v in _delays(*raw_j).items():
            row[f"person_delay_{j}_h" if k == "person_delay_h" else f"{k}_{j}"] = v
    for t, js in JUNCTION_TYPES.items():
        if len(js) == 1:
            continue                                   # = the junction's own columns
        d = _delays(sum((per[j][0] for j in js), []), sum(per[j][1] for j in js),
                    sum(per[j][2] for j in js))
        row.update({f"bus_delay_{t}": d["bus_delay"], f"car_delay_{t}": d["car_delay"],
                    f"person_delay_{t}_h": d["person_delay_h"]})
    return row


_GEO, _POLICIES = None, {}


def run_one(name, seed, policy=None, model_path=None, duration=7200.0, tag=None,
            obs_version="v1", scope="b1"):
    """One grid run with `policy` (callable obs -> action, fed `obs_version`
    observations) or `model_path` (version read from the model) at B1 (scope b1) or at
    all 9 junctions (all9); neither = C everywhere. Returns (row, line_rows) like
    c_experiments.run_job."""
    assert scope in SCOPES, scope
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
    with (rl_policy(policy, holder, obs_version, scope) if policy else contextlib.nullcontext()):
        row, lines, _, _ = gd.run_seed(seed, args, _GEO)
    row["wall_s"] = round(time.time() - tic, 1)
    raw = os.path.join(RESULTS, "raw", tag, f"seed_{seed}")
    route = os.path.join(gd.HERE, "routes", tag, f"routes_seed{seed}.rou.xml")
    row.update(junction_measures(raw, route, _GEO, args.warmup, args.warmup + args.duration))
    ctrls = holder.get("ctrls") or {}
    row["rl_shield_overrides"] = (sum(len(c.signal.shield_log) for c in ctrls.values())
                                  if ctrls else math.nan)
    row["rl_violations"] = (sum(len(c.signal.violations) for c in ctrls.values())
                            if ctrls else math.nan)
    row["obs_version"] = next(iter(ctrls.values())).obs_version if ctrls else ""
    row["scope"] = scope if ctrls else "C"
    for j, c in ctrls.items():
        row[f"rl_switches_{j}"] = c.switches
        row[f"rl_shield_{j}"] = len(c.signal.shield_log)
        row[f"rl_violations_{j}"] = len(c.signal.violations)
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
                         tag=f"{job['exp']}_{job['name']}".replace("@", "_at_"),
                         scope=job.get("scope", "b1"))
    return row, lines, []


SEEDS = {"final": EVAL_SEEDS, "val": VAL_SEEDS}


def experiment(what, scope):
    """Output prefix in results/: b1 keeps the old names, all9 gets its own files."""
    if scope == "b1":
        return f"rl_{what}"
    return "rl_all9" if what == "final" else f"rl_all9_{what}"


def plan(models, scope="b1", with_b1=False):
    """[(row name, model path or None, scope)]: C_all first, then the models."""
    out = [("C_all", None, "b1")]
    for label, path in models.items():
        if scope == "b1":
            out.append((label, path, "b1"))
        else:
            if with_b1:
                out.append((f"{label}@B1", path, "b1"))
            out.append((f"{label}@all9", path, "all9"))
    return out


def evaluate(what, models, workers, scope="b1", with_b1=False):
    """models: {label: model path}, run after C_all in the given order."""
    runs = plan(models, scope, with_b1)
    names = [n for n, _, _ in runs]
    seeds = SEEDS[what]
    jobs = [dict(name=n, seed=s, flow=ce.FLOW, model=m, controller="rl", scope=sc)
            for n, m, sc in runs for s in seeds]
    ce.N_WORKERS = workers
    exp = experiment(what, scope)
    df, _, _ = ce.run_all(jobs, exp, os.path.join(RESULTS, f"{exp}_runs.csv"),
                          os.path.join(RESULTS, f"{exp}_lines.csv"), fn=final_job)
    assert (df.backend == "libsumo").all()
    title = "Final evaluation" if what == "final" else "Validation"
    s = summarize(df, names, os.path.join(RESULTS, f"{exp}_summary.csv"),
                  f"{title}, seeds {seeds[0]}-{seeds[-1]}", per_junction=scope == "all9")
    if what == "val" and scope == "b1":
        rank(s)


def smoke(models, scope, seed, duration):
    """One headless run per model (no CSV): safety, teleports and switches per junction."""
    bad = False
    for name, path, sc in plan(models, scope)[1:]:
        tic = time.time()
        row, _ = run_one(name, seed, model_path=path, duration=duration,
                         tag=f"rl_smoke_{name}".replace("@", "_at_"), scope=sc)
        print(f"\n[smoke] {name}: seed {seed}, {duration:.0f} s after the warm-up, backend "
              f"{row['backend']}, obs {row['obs_version']}, {time.time() - tic:.0f} s wall")
        print(f"  safety violations {row['rl_violations']:.0f}, shield overrides "
              f"{row['rl_shield_overrides']:.0f}, teleports {row['teleports']:.0f}, "
              f"never inserted {row.get('never_inserted', math.nan)}")
        js = [j for j in TLS_IDS if f"rl_switches_{j}" in row]
        print("  switches / violations per junction: " + ", ".join(
            f"{j} {row[f'rl_switches_{j}']}/{row[f'rl_violations_{j}']}" for j in js))
        idle = [j for j in js if row[f"rl_switches_{j}"] == 0]
        if row["rl_violations"] or idle or row["backend"] != "libsumo":
            bad = True
            print(f"  PROBLEM: violations {row['rl_violations']}, junctions never switching {idle}")
    print("\nSMOKE", "FAILED" if bad else "OK")
    return 1 if bad else 0


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


def paired_t(diff):
    """t statistic of a paired difference and whether |t| exceeds the two-sided 95 %
    Student t quantile (the quantiles of the CIs, c_experiments.T975)."""
    x = pd.Series(diff).dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return math.nan, False
    t = x.mean() / (x.std(ddof=1) / math.sqrt(n))
    return t, bool(abs(t) > ce.T975.get(n - 1, 1.96))


def summarize(df, names, path, title, per_junction=False):
    cols = ce.SUMMARY_COLS + B1_COLS + ["rl_shield_overrides", "rl_violations"]
    if per_junction:
        cols = cols + [c for c in TYPE_COLS if c not in cols] + \
            [f"{m}_{j}" for j in TLS_IDS for m in ("bus_delay", "car_delay")] + \
            [f"person_delay_{j}_h" for j in TLS_IDS if j != "B1"]
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
                    r[f"{c}_diff_t"], r[f"{c}_diff_sig"] = paired_t(diff)
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
    print("\nPaired difference vs C_all (* = paired t test significant at 5 %):")
    diff_cols = ["bunch_last_total", "car_delay_total", "cross_delay", "person_delay_h"] + B1_COLS
    if per_junction:
        diff_cols += ["cv_last_mean", "bus_jdelay"] + [c for c in TYPE_COLS if c not in diff_cols]
    for c in diff_cols:
        print(f"{c:<20}" + "".join(
            f"{r[c + '_diff']:>11.2f}{'*' if r[c + '_diff_sig'] else ' '}+-{r[c + '_diff_ci95']:>5.2f}"
            for _, r in s.iterrows()
            if r["name"] != "C_all" and c + "_diff" in r))
    return s


def main():
    ap = argparse.ArgumentParser(description="Evaluate B1 policies with the step-4 measures")
    ap.add_argument("what", choices=["final", "val", "smoke"],
                    help="final: seeds 10000-10019; val: validation seeds 900-904; "
                         "smoke: one run per model, nothing written")
    ap.add_argument("--scope", default="b1", choices=SCOPES,
                    help="b1: the model at B1, C elsewhere (default); all9: a copy at every junction")
    ap.add_argument("--with-b1", action="store_true",
                    help="with --scope all9: also run each model at B1 only (<label>@B1)")
    ap.add_argument("--seed", type=int, default=VAL_SEEDS[0], help="smoke: seed (not 10000+)")
    ap.add_argument("--duration", type=float, default=1800.0, help="smoke: seconds after warm-up")
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
    if a.with_b1 and a.scope != "all9":
        ap.error("--with-b1 needs --scope all9")
    if a.what == "smoke":
        if not models or a.seed >= EVAL_SEEDS[0]:
            ap.error("smoke needs --model and a seed below 10000 (never an evaluation seed)")
        sys.exit(smoke(models, a.scope, a.seed, a.duration))
    evaluate(a.what, models, a.workers, a.scope, a.with_b1)


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
