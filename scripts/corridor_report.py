"""Compare all controllers on the corridor over the evaluation seeds.

Owner: Adem.

    python scripts/corridor_report.py                      # 5 controllers x 10 seeds, 1800 s each
    python scripts/corridor_report.py --seconds 300 --seeds 2   # quick smoke test

For each controller and each seed in config.yaml (evaluation.seeds) it runs
scripts/run.py headless into runs/corridor_report/<controller>_seed<N>/ (up to
4 runs at a time), then reads each run's summary.csv and writes:
- runs/corridor_report/results.csv  one row per run
- runs/corridor_report/table.md     mean +/- std per controller, change vs fixed_time in %
- runs/corridor_report/results.png  bar chart: mean waiting and ambulance trip time
Numbers are reported exactly as they come out.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window, only the PNG file
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONTROLLERS = ["fixed_time", "green_wave", "max_pressure", "flow", "orchestrated"]

# column in results.csv -> metric name in each run's summary.csv
METRICS = {
    "mean_waiting_s": "mean_waiting_per_vehicle_s",
    "mean_trip_s": "mean_trip_duration_s",
    "arrived": "arrived_vehicles",
    "amb_1_trip_s": "emergency_travel_s_amb_1",
    "amb_2_trip_s": "emergency_travel_s_amb_2",
    "decision_ms": "mean_decision_ms",
}


def run_one(controller: str, seed: int, seconds: int, out: Path) -> tuple[str, int, int]:
    """Run one simulation; returns (controller, seed, exit code)."""
    run_dir = out / f"{controller}_seed{seed}"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run.py"), "--controller", controller,
         "--scenario", "corridor", "--seed", str(seed), "--seconds", str(seconds),
         "--run-dir", str(run_dir)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        (out / f"{controller}_seed{seed}_error.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    return controller, seed, result.returncode


def read_summary(run_dir: Path) -> dict:
    """summary.csv (metric,value) -> the columns of results.csv; missing values stay empty."""
    path = run_dir / "summary.csv"
    if not path.is_file():
        return {}
    summary = dict(pd.read_csv(path).values)
    return {col: pd.to_numeric(summary.get(name), errors="coerce") for col, name in METRICS.items()}


def make_table(results: pd.DataFrame) -> str:
    """Markdown table: mean +/- std per controller, and change of the mean vs fixed_time."""
    stats = results.groupby("controller", sort=False)[list(METRICS)].agg(["mean", "std"])
    base = stats.loc["fixed_time"] if "fixed_time" in stats.index else None
    seeds = results.groupby("controller", sort=False)["seed"].nunique()
    header = "| controller | seeds | " + " | ".join(METRICS) + " |"
    lines = [header, "|" + "---|" * (len(METRICS) + 2)]
    for controller in stats.index:
        cells = []
        for col in METRICS:
            mean, std = stats.loc[controller, (col, "mean")], stats.loc[controller, (col, "std")]
            if pd.isna(mean):
                cells.append("-")
                continue
            cell = f"{mean:.2f} +/- {0 if pd.isna(std) else std:.2f}"
            if base is not None and controller != "fixed_time" and col != "decision_ms":
                ref = base[(col, "mean")]
                if pd.notna(ref) and ref != 0:
                    cell += f" ({100 * (mean - ref) / ref:+.1f}%)"
            cells.append(cell)
        lines.append(f"| {controller} | {seeds[controller]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_chart(results: pd.DataFrame, path: Path) -> None:
    """Two bar charts: mean waiting per vehicle, and ambulance trip times."""
    means = results.groupby("controller", sort=False)[["mean_waiting_s", "amb_1_trip_s", "amb_2_trip_s"]].mean()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    means["mean_waiting_s"].plot.bar(ax=left, color="tab:blue")
    left.set_title("Mean waiting per vehicle")
    left.set_ylabel("seconds")
    ambulance = means[["amb_1_trip_s", "amb_2_trip_s"]].dropna(axis=1, how="all")
    if not ambulance.empty:
        ambulance.plot.bar(ax=right, color=["tab:red", "tab:orange"][:ambulance.shape[1]])
    right.set_title("Ambulance trip time")
    right.set_ylabel("seconds")
    for ax in (left, right):
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    seeds = results["seed"].nunique()
    fig.suptitle(f"Corridor, {seeds} seeds (mean)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Compare controllers on the corridor over the evaluation seeds.")
    p.add_argument("--seconds", type=int, default=1800, help="simulated seconds per run (default 1800)")
    p.add_argument("--seeds", type=int, default=None, help="use only the first N seeds of evaluation.seeds")
    p.add_argument("--workers", type=int, default=4, help="runs at the same time (default 4)")
    p.add_argument("--out", default=str(ROOT / "runs" / "corridor_report"), help="output folder")
    args = p.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    seeds = config["evaluation"]["seeds"][: args.seeds]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s) for c in CONTROLLERS for s in seeds]
    print(f"{len(jobs)} runs ({len(CONTROLLERS)} controllers x {len(seeds)} seeds, {args.seconds} s each), "
          f"{args.workers} at a time -> {out}")

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for controller, seed, code in pool.map(lambda job: run_one(*job, args.seconds, out), jobs):
            print(f"  {controller:13s} seed {seed}: {'ok' if code == 0 else f'FAILED (exit {code})'}", flush=True)
            if code != 0:
                failed.append((controller, seed))

    rows = [{"controller": c, "seed": s, **read_summary(out / f"{c}_seed{s}")} for c, s in jobs
            if (c, s) not in failed]
    for row in rows:  # only the orchestrator times its decisions; the baselines would show 0
        if row["controller"] != "orchestrated":
            row["decision_ms"] = float("nan")
    results = pd.DataFrame(rows, columns=["controller", "seed", *METRICS])
    results.to_csv(out / "results.csv", index=False)
    table = make_table(results)
    (out / "table.md").write_text(
        f"# Corridor report: {len(seeds)} seeds, {args.seconds} s per run\n\n"
        "Mean +/- standard deviation over the seeds; (change of the mean vs fixed_time).\n"
        "Lower is better for every column except arrived. decision_ms only exists for orchestrated.\n\n"
        + table + "\n", encoding="utf-8")
    make_chart(results, out / "results.png")

    print("\n" + table)
    if failed:
        print(f"\nFAILED runs (see *_error.txt in {out}): {failed}")
    print(f"\nWritten: {out / 'results.csv'}, {out / 'table.md'}, {out / 'results.png'}")


if __name__ == "__main__":
    main()
