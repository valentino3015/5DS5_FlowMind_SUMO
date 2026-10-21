"""
Demand generator for the BO4 3x3 grid scenario (one route file per seed).

    python bo4_bus/gen_grid_routes.py --seed 10000
    python bo4_bus/gen_grid_routes.py --seed 10000 --entry-flow 675 --dispatch-sigma 60

Cars
----
Each of the 12 fringe entry links has its own Poisson process (exponential gaps,
numpy RNG seeded with --seed) with rate --entry-flow veh/h. Destination per car:
the exit straight across the grid with probability --straight-share, otherwise one
of the other fringe exits uniformly (never the exit next to its own entry, i.e. no
U-turn). Cars are written as <trip>s; SUMO picks the fastest route on insertion,
so cars turn at whichever junction the router chooses.

Buses
-----
Lines L1 (E-W, middle row: L1_EB, L1_WB) and L2 (N-S, middle column: L2_NB,
L2_SB), stops defined in nets/grid/build_grid.py. Per line direction:
    t_k = first + k*H + clip(N(0, sigma), -0.45 H, +0.45 H)   (order preserved)
Stop durations in the file are a placeholder (DWELL_PLACEHOLDER): the real dwell is
set in closed loop by Grid_Demo.py on arrival (see dwell_time() there and
ASSUMPTIONS.md). Running this route file without Grid_Demo.py is not meaningful.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "nets", "grid"))
from build_grid import LINES  # noqa: E402

CAR_OCCUPANCY = 1.3
DWELL_PLACEHOLDER = 3600   # s, overridden by Grid_Demo.py (vehicle.resume)

# fringe node -> (entry edge, exit edge, opposite fringe node)
FRINGE = {}
for i, (c, r) in enumerate(zip("ABC", "012")):
    FRINGE[f"left{i}"] = (f"left{i}A{i}", f"A{i}left{i}", f"right{i}")
    FRINGE[f"right{i}"] = (f"right{i}C{i}", f"C{i}right{i}", f"left{i}")
    FRINGE[f"bottom{i}"] = (f"bottom{i}{c}0", f"{c}0bottom{i}", f"top{i}")
    FRINGE[f"top{i}"] = (f"top{i}{c}2", f"{c}2top{i}", f"bottom{i}")


def poisson_departures(rng, rate_vph, t_end):
    if rate_vph <= 0:
        return []
    mean_gap = 3600.0 / rate_vph
    times, t = [], rng.exponential(mean_gap)
    while t < t_end:
        times.append(t)
        t += rng.exponential(mean_gap)
    return times


def peak_multiplier(t, factor, start, duration, period):
    """Demand multiplier at time t: `factor` inside peaks [start + k*period, + duration), else 1."""
    if factor == 1.0 or duration <= 0 or t < start:
        return 1.0
    return factor if (t - start) % period < duration else 1.0


def profiled_departures(rng, rate_vph, t_end, factor, start, duration, period):
    """Non-homogeneous Poisson process by thinning: rate(t) = rate_vph * peak_multiplier(t)."""
    top = max(1.0, factor)
    cand = poisson_departures(rng, rate_vph * top, t_end)
    keep = rng.random(len(cand)) < [peak_multiplier(t, factor, start, duration, period) / top
                                    for t in cand]
    return [t for t, k in zip(cand, keep) if k]


def dispatch_times(rng, headway, sigma, t_end, first=30.0):
    times, k = [], 0
    while first + k * headway < t_end:
        eps = float(np.clip(rng.normal(0.0, sigma), -0.45 * headway, 0.45 * headway))
        times.append(max(0.0, first + k * headway + eps))
        k += 1
    return times


def generate(seed, out_path, t_end=7800.0, entry_flow=675.0, straight_share=0.6,
             peak_factor=1.0, peak_start=1200.0, peak_duration=900.0, peak_period=3600.0,
             headway=180.0, dispatch_sigma=60.0, occ_min=20, occ_max=60):
    rng = np.random.default_rng(seed)
    vehicles = []

    for src, (entry, _, opposite) in FRINGE.items():
        others = [n for n in FRINGE if n not in (src, opposite)]
        dests = [opposite] + others
        probs = [straight_share] + [(1 - straight_share) / len(others)] * len(others)
        deps = profiled_departures(rng, entry_flow, t_end, peak_factor, peak_start,
                                   peak_duration, peak_period)
        picks = rng.choice(len(dests), p=probs, size=len(deps))
        for i, (t, j) in enumerate(zip(deps, picks)):
            vehicles.append((t, f'    <trip id="car_{src}_{i}" type="car" depart="{t:.2f}" '
                                f'from="{entry}" to="{FRINGE[dests[j]][1]}" '
                                f'departLane="best" departSpeed="max"/>\n'))

    for d, spec in LINES.items():
        for k, t in enumerate(dispatch_times(rng, headway, dispatch_sigma, t_end)):
            occ = int(rng.integers(occ_min, occ_max + 1))
            stops = "".join(f'        <stop busStop="{s}" duration="{DWELL_PLACEHOLDER}"/>\n'
                            for s in spec["stops"])
            vehicles.append((t, f'    <vehicle id="{d}_{k:02d}" type="bus" depart="{t:.2f}" '
                                f'departLane="0" departSpeed="max">\n'
                                f'        <route edges="{" ".join(spec["edges"])}"/>\n'
                                f'{stops}'
                                f'        <param key="line" value="{d}"/>\n'
                                f'        <param key="occupancy" value="{occ}"/>\n'
                                f'    </vehicle>\n'))

    vehicles.sort(key=lambda v: v[0])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"<!-- generated by bo4_bus/gen_grid_routes.py seed={seed} entry_flow={entry_flow} "
                f"straight_share={straight_share} peak={peak_factor}x{peak_duration}s/{peak_period}s@{peak_start} H={headway} sigma={dispatch_sigma} -->\n")
        f.write("<routes>\n")
        f.write('    <vType id="car" vClass="passenger" length="5" accel="2.6" decel="4.5" '
                'sigma="0.5"/>\n')
        f.write('    <vType id="bus" vClass="bus" length="12" accel="1.2" decel="4.0" '
                'sigma="0.5" maxSpeed="13.89" speedFactor="1" speedDev="0"/>\n')
        for _, xml in vehicles:
            f.write(xml)
        f.write("</routes>\n")
    return out_path


def add_args(ap):
    ap.add_argument("--entry-flow", type=float, default=675.0,
                    help="car flow per fringe entry link [veh/h]")
    ap.add_argument("--straight-share", type=float, default=0.6,
                    help="share of cars heading to the exit straight across")
    ap.add_argument("--peak-factor", type=float, default=1.0,
                    help="car demand multiplier during peaks (1 = stationary demand)")
    ap.add_argument("--peak-start", type=float, default=1200.0, help="first peak start [s]")
    ap.add_argument("--peak-duration", type=float, default=900.0, help="peak length [s]")
    ap.add_argument("--peak-period", type=float, default=3600.0, help="peak repetition [s]")
    ap.add_argument("--headway", type=float, default=180.0, help="target bus headway H [s]")
    ap.add_argument("--dispatch-sigma", type=float, default=60.0, help="dispatch jitter std [s]")
    return ap


def gen_kwargs(args):
    return dict(entry_flow=args.entry_flow, straight_share=args.straight_share,
                peak_factor=args.peak_factor, peak_start=args.peak_start,
                peak_duration=args.peak_duration, peak_period=args.peak_period,
                headway=args.headway, dispatch_sigma=args.dispatch_sigma)


def main():
    ap = add_args(argparse.ArgumentParser(description="BO4 grid demand generator"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--t-end", type=float, default=7800.0, help="last departure [s]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "routes", "grid", f"routes_seed{args.seed}.rou.xml")
    generate(args.seed, out, t_end=args.t_end, **gen_kwargs(args))
    print("wrote", out)


if __name__ == "__main__":
    main()
