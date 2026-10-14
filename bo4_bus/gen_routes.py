"""
Demand generator for the BO4 bus scenario (one route file per seed).

    python bo4_bus/gen_routes.py --seed 1
    python bo4_bus/gen_routes.py --seed 1 --art-flow 1000 --headway 240 --out my.rou.xml

Cars
----
Every car is written as an explicit <vehicle> whose departure times come from
a Poisson process (exponential inter-arrival times, numpy RNG seeded with
--seed), so the demand is fully determined by the seed. Each approach has its
own process; turn choice is a categorical draw per car.

Bus line L1 (EB: W->E, WB: E->W)
--------------------------------
Dispatch:  t_k = first + k*H + eps_k,   eps_k ~ Normal(0, sigma) clipped to
           [-0.45 H, +0.45 H], so dispatch order is always preserved.

Dwell at each of its two stops (per bus, per stop):
           duration = t_fixed + t_board * B,   B ~ Poisson(lambda * gap)
           gap      = time since the previous bus of the same direction at that
                      stop (estimated, see below)
           lambda   = (mean_dwell - t_fixed) / (t_board * H)   [pax/s]
           Defaults t_fixed=10 s, t_board=2.5 s/pax, mean_dwell=25 s, H=240 s
           -> E[B] = 6 pax, lambda = 0.025 pax/s.
A bus that runs late has a larger gap, picks up more passengers, dwells longer
and falls further behind (the follower gets a shorter gap and catches up):
this is the classic bunching mechanism.

The dwell is fixed in the route file (open loop), so the gap has to be
estimated before simulation:
    upstream stop   : gap = t_k - t_{k-1}  (dispatch gap; free-flow running
                      time to the stop is the same for every bus)
    downstream stop : gap = (t_k + dwell_up_k) - (t_{k-1} + dwell_up_{k-1})
                      i.e. it carries the dwell variability of the first stop.
Signal delay is NOT fed back into the dwell (no closed loop). See ASSUMPTIONS.md.
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

CAR_OCCUPANCY = 1.3     # persons per car (constant, used by the metrics)
BUS_LINE = "L1"

# approach edge -> (straight, left, right) destination edges
TURNS = {
    "W2C": ("C2E", "C2N", "C2S"),
    "E2C": ("C2W", "C2S", "C2N"),
    "N2C": ("C2S", "C2E", "C2W"),
    "S2C": ("C2N", "C2W", "C2E"),
}
ARTERIAL = ("W2C", "E2C")

BUS_DIRS = {
    "EB": (("W2C", "C2E"), ("EB_up", "EB_down")),
    "WB": (("E2C", "C2W"), ("WB_up", "WB_down")),
}


def poisson_departures(rng, rate_vph, t_end):
    """Departure times of a Poisson process with rate_vph veh/h on [0, t_end)."""
    if rate_vph <= 0:
        return []
    mean_gap = 3600.0 / rate_vph
    times, t = [], rng.exponential(mean_gap)
    while t < t_end:
        times.append(t)
        t += rng.exponential(mean_gap)
    return times


def dispatch_times(rng, headway, sigma, t_end, first=60.0):
    """t_k = first + k*H + clip(N(0, sigma), +-0.45 H) for all t_k < t_end."""
    times, k = [], 0
    while first + k * headway < t_end:
        eps = float(np.clip(rng.normal(0.0, sigma), -0.45 * headway, 0.45 * headway))
        times.append(max(0.0, first + k * headway + eps))
        k += 1
    return times


def dwell_times(rng, gaps, lam, t_fixed, t_board):
    """duration = t_fixed + t_board * Poisson(lam * gap) for each gap."""
    boardings = rng.poisson(lam * np.asarray(gaps, dtype=float))
    return [t_fixed + t_board * int(b) for b in boardings], [int(b) for b in boardings]


def generate(seed, out_path, t_end=4200.0, art_flow=1100.0, cross_flow=450.0,
             art_left=0.05, art_right=0.10, cross_left=0.10, cross_right=0.15,
             headway=240.0, dispatch_sigma=45.0, mean_dwell=25.0, t_fixed=10.0,
             t_board=2.5, occ_min=20, occ_max=60):
    rng = np.random.default_rng(seed)
    vehicles = []  # (depart, xml)

    # ---- cars -------------------------------------------------------------
    for src, (straight, left, right) in TURNS.items():
        if src in ARTERIAL:
            rate, pl, pr = art_flow, art_left, art_right
        else:
            rate, pl, pr = cross_flow, cross_left, cross_right
        deps = poisson_departures(rng, rate, t_end)
        dests = rng.choice([straight, left, right], p=[1 - pl - pr, pl, pr], size=len(deps))
        for i, (t, dst) in enumerate(zip(deps, dests)):
            vehicles.append((t, f'    <vehicle id="car_{src}_{i}" type="car" depart="{t:.2f}" '
                                f'departLane="best" departSpeed="max">\n'
                                f'        <route edges="{src} {dst}"/>\n'
                                f'    </vehicle>\n'))

    # ---- buses ------------------------------------------------------------
    lam = (mean_dwell - t_fixed) / (t_board * headway)
    for d, (edges, (stop_up, stop_down)) in BUS_DIRS.items():
        line = f"{BUS_LINE}_{d}"
        t_disp = dispatch_times(rng, headway, dispatch_sigma, t_end)
        gaps_up = [headway] + list(np.diff(t_disp))
        dwell_up, b_up = dwell_times(rng, gaps_up, lam, t_fixed, t_board)
        arr_down = np.asarray(t_disp) + np.asarray(dwell_up)
        gaps_down = [headway] + list(np.diff(arr_down))
        dwell_down, b_down = dwell_times(rng, gaps_down, lam, t_fixed, t_board)
        for k, t in enumerate(t_disp):
            occ = int(rng.integers(occ_min, occ_max + 1))
            vehicles.append((t, f'    <vehicle id="{line}_{k:02d}" type="bus" depart="{t:.2f}" '
                                f'departLane="0" departSpeed="max">\n'
                                f'        <route edges="{" ".join(edges)}"/>\n'
                                f'        <stop busStop="{stop_up}" duration="{dwell_up[k]:.1f}"/>\n'
                                f'        <stop busStop="{stop_down}" duration="{dwell_down[k]:.1f}"/>\n'
                                f'        <param key="line" value="{line}"/>\n'
                                f'        <param key="occupancy" value="{occ}"/>\n'
                                f'        <param key="boardings" value="{b_up[k]} {b_down[k]}"/>\n'
                                f'    </vehicle>\n'))

    vehicles.sort(key=lambda v: v[0])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"<!-- generated by bo4_bus/gen_routes.py seed={seed} art_flow={art_flow} "
                f"cross_flow={cross_flow} H={headway} sigma={dispatch_sigma} "
                f"mean_dwell={mean_dwell} lambda={lam:.5f} -->\n")
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
    ap.add_argument("--art-flow", type=float, default=1100.0, help="veh/h per arterial direction")
    ap.add_argument("--cross-flow", type=float, default=450.0, help="veh/h per cross direction")
    ap.add_argument("--art-left", type=float, default=0.05)
    ap.add_argument("--art-right", type=float, default=0.10)
    ap.add_argument("--cross-left", type=float, default=0.10)
    ap.add_argument("--cross-right", type=float, default=0.15)
    ap.add_argument("--headway", type=float, default=240.0, help="target bus headway H [s]")
    ap.add_argument("--dispatch-sigma", type=float, default=45.0, help="dispatch jitter std [s]")
    ap.add_argument("--mean-dwell", type=float, default=25.0, help="target mean dwell [s]")
    ap.add_argument("--board-time", type=float, default=2.5, help="s per boarding")
    return ap


def gen_kwargs(args):
    return dict(art_flow=args.art_flow, cross_flow=args.cross_flow,
                art_left=args.art_left, art_right=args.art_right,
                cross_left=args.cross_left, cross_right=args.cross_right,
                headway=args.headway, dispatch_sigma=args.dispatch_sigma,
                mean_dwell=args.mean_dwell, t_board=args.board_time)


def main():
    ap = add_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--t-end", type=float, default=4200.0, help="last departure [s]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "routes", f"routes_seed{args.seed}.rou.xml")
    generate(args.seed, out, t_end=args.t_end, **gen_kwargs(args))
    print("wrote", out)


if __name__ == "__main__":
    main()
