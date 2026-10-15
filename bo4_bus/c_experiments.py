"""
Step 4 experiments for controller C (conditional TSP) on the BO4 3x3 grid.

    python bo4_bus/c_experiments.py tune      # grid search L x Q x MAX_EXT, training seeds 0-9
    python bo4_bus/c_experiments.py stress    # fixed / B / C_tuned at 600, 675, 725 veh/h, seeds 0-4
    python bo4_bus/c_experiments.py final     # fixed / B / C_default / C_tuned, seeds 10000-10019
    python bo4_bus/c_experiments.py figure    # results/cv_along_line.png from the final runs
    python bo4_bus/c_experiments.py all       # tune, stress, final, figure (C_TUNED must be set)

Every run is one Grid_Demo.run_seed() call (headless libsumo), several runs in
parallel worker processes. Raw SUMO outputs and route files of these runs are
deleted after their metrics are computed. Outputs (results/):
    c_tuning_runs.csv / c_tuning_summary.csv
    c_stress_runs.csv / c_stress_summary.csv
    c_final_runs.csv / c_final_lines.csv / c_final_decisions.csv / c_final_summary.csv
    cv_along_line.png

Tuning objective (on training seeds only): minimise the number of bunched
headways at the last stop (summed over the 4 line directions, mean over seeds)
subject to mean car delay (timeLoss + departDelay) <= 1.03 x fixed-time on the
same seeds. Ties: lower bus junction delay.
"""
import argparse
import itertools
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Grid_Demo as gd  # noqa: E402
import rule_tsp  # noqa: E402
from build_grid import LINES  # noqa: E402

RESULTS = gd.RESULTS
TRAIN_SEEDS = list(range(10))          # tuning / stress only, never 10000+
STRESS_SEEDS = list(range(5))
EVAL_SEEDS = list(range(10000, 10020))
FLOW = 675.0                           # veh/h per entry, scenario default
STRESS_FLOWS = [600.0, 675.0, 725.0]

GRID = {"late_thr": [0.0, 20.0, 40.0], "queue_thr": [8, 12, 15], "max_ext": [10.0, 15.0, 20.0]}
CAR_DELAY_TOL = 0.03                   # C may be at most 3 % worse than fixed-time

DEFAULT_PARAMS = {"late_thr": rule_tsp.LATE_THR, "queue_thr": rule_tsp.QUEUE_THR,
                  "max_ext": rule_tsp.MAX_EXT}
# Tuning result (results/c_tuning_summary.csv): the top 3 feasible settings are within 0.5
# last-stop bunched headways (s.e. ~1 per setting, i.e. tied). Strict #1 (L0 Q12 E20) had a
# breakdown on training seed 3 (55 never inserted); L0 Q15 E20 has none on any training seed
# and the lowest bus junction delay of the three -> chosen.
C_TUNED = {"late_thr": 0.0, "queue_thr": 15, "max_ext": 20.0}

CONFIGS = {   # name -> (Grid_Demo controller, C parameters)
    "fixed": ("fixed", DEFAULT_PARAMS),
    "B": ("tsp_all", DEFAULT_PARAMS),
    "C_default": ("tsp_cond", DEFAULT_PARAMS),
}

N_WORKERS = 10
T975 = {4: 2.776, 9: 2.262, 19: 2.093}   # Student t quantiles for 95 % CIs (df = n - 1)

SUMMARY_COLS = ["bus_jdelay", "bus_tt", "bunch_last_total", "bunch_total", "cv_last_mean",
                "car_delay_total", "car_timeloss", "cross_delay", "bus_appr_delay",
                "person_delay_h", "max_queue_any", "spill_longest_s", "never_inserted",
                "teleports", "tsp_requests", "tsp_granted", "tsp_refused", "refused_queue",
                "queue_rule_hits"]


# --------------------------------------------------------------------------
# one run (worker process)
# --------------------------------------------------------------------------
_GEO = None


def run_job(job):
    """job: dict(exp, name, controller, seed, flow, late_thr, queue_thr, max_ext).
    Returns (row, line_rows, decisions) annotated with the job's identity."""
    global _GEO
    if _GEO is None:
        _GEO = gd.build_geo()
    tag = f"c_{job['exp']}_{job['name']}_f{int(job['flow'])}"
    args = gd.parse_args(["--controller", job["controller"], "--seeds", str(job["seed"]),
                          "--tag", tag, "--entry-flow", str(job["flow"]),
                          "--late-thr", str(job["late_thr"]), "--queue-thr", str(job["queue_thr"]),
                          "--max-ext", str(job["max_ext"])])
    tic = time.time()
    row, lines, _, decisions = gd.run_seed(job["seed"], args, _GEO)
    row["wall_s"] = round(time.time() - tic, 1)
    shutil.rmtree(os.path.join(RESULTS, "raw", tag, f"seed_{job['seed']}"), ignore_errors=True)
    route = os.path.join(gd.HERE, "routes", tag, f"routes_seed{job['seed']}.rou.xml")
    if os.path.exists(route):
        os.remove(route)
    ident = {k: job[k] for k in ("name", "flow", "late_thr", "queue_thr", "max_ext")}
    row = {**ident, **row}
    row["bunch_last_total"] = sum(row[f"bunch_last_{d}"] for d in LINES)
    row["cv_last_mean"] = sum(row[f"cv_last_{d}"] for d in LINES) / len(LINES)
    row["bus_tt"] = sum(row[f"tt_{d}"] for d in LINES) / len(LINES)
    lines = [{**ident, **r} for r in lines]
    decisions = [{**ident, **r} for r in decisions]
    return row, lines, decisions


def run_all(jobs, exp):
    rows, lines, decisions = [], [], []
    tic = time.time()
    with ProcessPoolExecutor(N_WORKERS) as ex:
        futs = [ex.submit(run_job, dict(j, exp=exp)) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            r, l, d = f.result()
            assert r["backend"] == "libsumo", "official results need headless libsumo"
            rows.append(r)
            lines += l
            decisions += d
            print(f"[{exp}] {k}/{len(jobs)} {r['name']} flow={r['flow']:.0f} seed={r['seed']} "
                  f"({r['wall_s']} s, {time.time() - tic:.0f} s elapsed)", flush=True)
    order = ["name", "flow", "seed"]
    return (pd.DataFrame(rows).sort_values(order), pd.DataFrame(lines).sort_values(order),
            pd.DataFrame(decisions))


def job(name, seed, flow=FLOW, params=None):
    controller, base = CONFIGS[name] if name in CONFIGS else ("tsp_cond", None)
    return dict(name=name, controller=controller, seed=seed, flow=flow, **(params or base))


def tuned():
    if C_TUNED is None:
        sys.exit("set C_TUNED in c_experiments.py from results/c_tuning_summary.csv first")
    CONFIGS["C_tuned"] = ("tsp_cond", C_TUNED)
    return "C_tuned"


def ci95(x):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 2:
        return x.mean(), math.nan
    return x.mean(), T975.get(n - 1, 1.96) * x.std(ddof=1) / math.sqrt(n)


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def cfg_name(late, q, ext):
    return f"C_L{late:.0f}_Q{q}_E{ext:.0f}"


def tune():
    jobs = [job("fixed", s) for s in TRAIN_SEEDS]
    for late, q, ext in itertools.product(GRID["late_thr"], GRID["queue_thr"], GRID["max_ext"]):
        p = {"late_thr": late, "queue_thr": q, "max_ext": ext}
        jobs += [job(cfg_name(late, q, ext), s, params=p) for s in TRAIN_SEEDS]
    df, _, _ = run_all(jobs, "tune")
    df.to_csv(os.path.join(RESULTS, "c_tuning_runs.csv"), index=False)

    fixed_car = df[df.name == "fixed"].car_delay_total.mean()
    cols = ["bunch_last_total", "car_delay_total", "bus_jdelay", "cross_delay", "bunch_total",
            "cv_last_mean", "tsp_granted", "refused_queue", "never_inserted", "teleports"]
    s = df.groupby(["name", "late_thr", "queue_thr", "max_ext"])[cols].mean()
    s["bunch_last_se"] = df.groupby(["name", "late_thr", "queue_thr", "max_ext"]
                                    ).bunch_last_total.sem()
    s = s.reset_index()
    s["car_vs_fixed_pct"] = 100 * (s.car_delay_total / fixed_car - 1)
    s["feasible"] = s.car_delay_total <= (1 + CAR_DELAY_TOL) * fixed_car
    s.loc[s.name == "fixed", "feasible"] = False
    s["is_default"] = ((s.late_thr == DEFAULT_PARAMS["late_thr"]) & (s.queue_thr == DEFAULT_PARAMS["queue_thr"])
                       & (s.max_ext == DEFAULT_PARAMS["max_ext"]) & (s.name != "fixed"))
    s = s.sort_values(["feasible", "bunch_last_total", "bus_jdelay"], ascending=[False, True, True])
    s.to_csv(os.path.join(RESULTS, "c_tuning_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"\nfixed-time on training seeds: car delay {fixed_car:.2f} s, "
          f"last-stop bunching {df[df.name == 'fixed'].bunch_last_total.mean():.2f}")
    print("\nTop 5 feasible settings:")
    print(s[s.feasible].head(5).round(2).to_string(index=False))
    print("\nC_default:")
    print(s[s.is_default].round(2).to_string(index=False))
    print("\nAll settings:")
    print(s.round(2).to_string(index=False))


def stress():
    names = ["fixed", "B", tuned()]
    jobs = [job(n, s, flow=f) for f in STRESS_FLOWS for n in names for s in STRESS_SEEDS]
    df, _, _ = run_all(jobs, "stress")
    df.to_csv(os.path.join(RESULTS, "c_stress_runs.csv"), index=False)
    cols = ["bus_jdelay", "bunch_last_total", "car_delay_total", "cross_delay", "bus_appr_delay",
            "never_inserted", "teleports", "tsp_requests", "tsp_granted", "refused_queue",
            "queue_rule_hits"]
    s = df.groupby(["flow", "name"], sort=False)[cols].mean().reset_index()
    s["name"] = pd.Categorical(s.name, names)
    s = s.sort_values(["flow", "name"])
    s.to_csv(os.path.join(RESULTS, "c_stress_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\nStress test (mean over seeds 0-4):")
    print(s.round(2).to_string(index=False))


def final():
    names = ["fixed", "B", "C_default", tuned()]
    jobs = [job(n, s) for n in names for s in EVAL_SEEDS]
    df, dl, dd = run_all(jobs, "final")
    df.to_csv(os.path.join(RESULTS, "c_final_runs.csv"), index=False)
    dl.to_csv(os.path.join(RESULTS, "c_final_lines.csv"), index=False)
    dd.to_csv(os.path.join(RESULTS, "c_final_decisions.csv"), index=False)
    out = []
    for n in names:
        sub = df[df.name == n]
        r = {"name": n, "n_seeds": len(sub)}
        for c in SUMMARY_COLS:
            if c in sub:
                m, h = ci95(sub[c])
                r[c], r[f"{c}_ci95"] = m, h
        out.append(r)
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(RESULTS, "c_final_summary.csv"), index=False)
    print("\nFinal evaluation, seeds 10000-10019 (mean +- 95 % CI half-width):")
    print(f"{'metric':<18}" + "".join(f"{n:>20}" for n in names))
    for c in SUMMARY_COLS:
        if c in s:
            print(f"{c:<18}" + "".join(
                f"{r[c]:>12.2f} +-{r[c + '_ci95']:>5.2f}" if not pd.isna(r[c]) else f"{'-':>20}"
                for _, r in s.iterrows()))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dl = pd.read_csv(os.path.join(RESULTS, "c_final_lines.csv"))
    shown = [("fixed", "fixed-time", "#6b6b6b"), ("B", "B (unconditional)", "#d1495b"),
             ("C_tuned", "C tuned (conditional)", "#00798c")]
    fig, axes = plt.subplots(1, len(LINES), figsize=(13, 3.6), sharey=True)
    for ax, d in zip(axes, LINES):
        for name, label, color in shown:
            g = dl[(dl.name == name) & (dl.line == d)].groupby("stop_idx").cv
            m = g.mean()
            h = g.apply(lambda x: ci95(x)[1])
            ax.plot(m.index, m.values, marker="o", color=color, label=label)
            ax.fill_between(m.index, m - h, m + h, color=color, alpha=0.15, linewidth=0)
        ax.set_title(d)
        ax.set_xlabel("stop index")
        ax.set_xticks(range(4))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("headway CV")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Headway CV along the line (seeds {EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}, "
                 "mean and 95 % CI)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(RESULTS, "cv_along_line.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description="BO4 step 4: tune and stress-test controller C")
    ap.add_argument("what", choices=["tune", "stress", "final", "figure", "all"])
    a = ap.parse_args()
    steps = [tune, stress, final, figure] if a.what == "all" else [globals()[a.what]]
    for step in steps:
        step()


if __name__ == "__main__":
    main()
