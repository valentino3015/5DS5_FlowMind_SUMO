"""Headway CV along each line: fixed-time, C at all 9 lights, and agent F at B1 only and
at all 9 lights.

Reads results/c_final_lines.csv (fixed-time) and results/rl_all9_lines.csv (C_all and
every "<label>@B1" / "<label>@all9" row written by `evaluate.py final --scope all9`), and
writes results/cv_along_line_all9.png. Does not rerun any simulation and does not touch
the existing PNGs. Several models (F_s0, F_s1, ...) are pooled per scope with
--pool, otherwise each gets its own curve.

    python sandbox/bo4_bus/rl/plot_cv_all9.py [--pool]
    python sandbox/bo4_bus/rl/plot_cv_all9.py --lines results/rl_all9_val_lines.csv --out x.png

Fixed-time is drawn only when its seeds are the same as those of the RL file.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
T975 = {4: 2.776, 9: 2.262, 19: 2.093}   # as c_experiments.T975


def ci95(x):
    x = pd.Series(x).dropna()
    n = len(x)
    return x.mean(), (T975.get(n - 1, 1.96) * x.std(ddof=1) / n ** 0.5 if n > 1 else 0.0)


def series(rl, fixed, pool):
    """[(rows, label, colour)] in drawing order."""
    out = []
    if fixed is not None:
        out.append((fixed[fixed.name == "fixed"], "fixed-time (all 9)", "#9a9a9a"))
    out.append((rl[rl.name == "C_all"], "C rules (all 9)", "#00798c"))
    shades = {"B1": ["#edae49", "#d1495b", "#a4161a", "#f3722c"],
              "all9": ["#2060cc", "#7b2cbf", "#3a0ca3", "#4cc9f0"]}
    text = {"B1": "at B1, C elsewhere", "all9": "at all 9"}
    for scope in ("B1", "all9"):
        names = [n for n in rl.name.unique() if str(n).endswith(f"@{scope}")]
        if not names:
            continue
        if pool:
            # one curve per agent (label before "_s<seed>", e.g. G_s0, G_s1 -> G): per
            # (seed, line, stop) the mean over that agent's models
            agents = {}
            for n in names:
                agents.setdefault(n.split("@")[0].rsplit("_s", 1)[0], []).append(n)
            for i, (agent, ns) in enumerate(agents.items()):
                sub = rl[rl.name.isin(ns)].groupby(["seed", "line", "stop_idx"],
                                                   as_index=False).cv.mean()
                label = f"PPO {agent} {text[scope]}" + (
                    f" (mean of {len(ns)} training seeds)" if len(ns) > 1 else "")
                out.append((sub, label, shades[scope][i % 4]))
        else:
            for i, n in enumerate(names):
                out.append((rl[rl.name == n], f"PPO {n.split('@')[0]} {text[scope]}",
                            shades[scope][i % 4]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default=os.path.join(RESULTS, "rl_all9_lines.csv"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "cv_along_line_all9.png"))
    ap.add_argument("--pool", action="store_true", help="pool several models per scope")
    a = ap.parse_args()

    rl = pd.read_csv(a.lines)
    fixed = pd.read_csv(os.path.join(RESULTS, "c_final_lines.csv"))
    fixed = fixed[fixed.name == "fixed"]
    rl_seeds = set(rl.seed)
    if set(fixed.seed) != rl_seeds:
        print(f"fixed-time seeds differ from {os.path.basename(a.lines)}: not drawn")
        fixed = None
    draw = series(rl, fixed, a.pool)
    print("plotting:", [label for _, label, _ in draw])

    lines = sorted(rl.line.unique())
    fig, axes = plt.subplots(1, len(lines), figsize=(14, 3.8), sharey=True)
    for ax, d in zip(axes, lines):
        for df, label, color in draw:
            g = df[df.line == d].groupby("stop_idx").cv
            if g.ngroups == 0:
                continue
            m = g.mean()
            h = g.apply(lambda x: ci95(x)[1])
            ax.plot(m.index, m.values, marker="o", color=color, label=label)
            ax.fill_between(m.index, m - h, m + h, color=color, alpha=0.12, linewidth=0)
        ax.set_title(d)
        ax.set_xlabel("stop index")
        ax.set_xticks(sorted(rl.stop_idx.unique()))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("headway CV (lower = more regular)")
    axes[0].legend(frameon=False, fontsize=8)
    seeds = sorted(rl_seeds)
    fig.suptitle(f"Headway CV along the line, seeds {seeds[0]}-{seeds[-1]} "
                 f"({len(seeds)} seeds, mean and 95 % CI)", fontsize=10)
    fig.tight_layout()
    if os.path.exists(a.out) and os.path.abspath(a.out) != os.path.abspath(
            os.path.join(RESULTS, "cv_along_line_all9.png")):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    fig.savefig(a.out, dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()
