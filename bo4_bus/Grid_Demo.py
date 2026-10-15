"""
BO4 3x3 grid scenario - fixed-time baseline with closed-loop bus dwell.

    python bo4_bus/Grid_Demo.py                         # evaluation seeds 10000-10004
    python bo4_bus/Grid_Demo.py --seeds 10000 --gui
    python bo4_bus/Grid_Demo.py --seeds 1 --duration 1800 --tag grid_calib   # short run

For each seed: generate routes (gen_grid_routes.py), run SUMO for warm-up (600 s)
+ measurement (7200 s), then compute metrics (bus_metrics.py) over the
measurement window. Writes results/<tag>.csv (one row per seed),
results/<tag>_lines.csv (per line direction and stop) and results/<tag>_queues.csv
(per signal approach).

Closed-loop dwell (every stop): when a bus starts a stop at time t,
    gap  = t - (arrival of the previous bus of the same direction at that stop)
           (H for the first bus at a stop)
    B    ~ Poisson(lambda * gap),  lambda = (MEAN_DWELL - T_FIXED) / (T_BOARD * H)
    dwell = T_FIXED + T_BOARD * B          (10 + 2.5 B, mean 25 s at gap = H)
and the bus is resumed once the dwell has elapsed. The RNG is seeded per run.

Signal control is plugged in per junction: controllers[tls_id].act(t, sim) is
called after every simulation step (sim = the traci/libsumo module);
FixedTimeController does nothing.
"""
import argparse
import time
import math
import os
import sys

if "SUMO_HOME" not in os.environ:
    for _p in (r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Program Files\Eclipse\Sumo"):
        if os.path.isdir(_p):
            os.environ["SUMO_HOME"] = _p
            break

import numpy as np
import pandas as pd
import sumolib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "nets", "grid"))
import bus_metrics as bm  # noqa: E402
import gen_grid_routes as gen  # noqa: E402
from build_grid import LINES, NET, STOPS_ADD, TLS_IDS, SPEED  # noqa: E402
from Bus_Demo import import_traci  # noqa: E402

RESULTS = os.path.join(HERE, "results")

T_FIXED, T_BOARD, MEAN_DWELL = 10.0, 2.5, 25.0
SAMPLE_DT = 5           # s, queue sampling period
QUEUE_SPEED = 1.39      # m/s: slower vehicles count as queued
QUEUE_GAP = 25.0        # m, max gap between consecutive queued vehicles
SPILL_POS = 20.0        # m: a halting vehicle this close to a link's upstream end = spillback


class FixedTimeController:
    """Baseline: leaves the static program of its junction untouched."""

    def __init__(self, tls_id):
        self.tls_id = tls_id

    def reset(self, sim):
        pass

    def act(self, t, sim):
        pass


def dwell_time(rng, gap, lam):
    """Closed-loop dwell: T_FIXED + T_BOARD * Poisson(lam * gap). Returns (dwell, boardings)."""
    b = int(rng.poisson(lam * max(gap, 0.0)))
    return T_FIXED + T_BOARD * b, b


# --------------------------------------------------------------------------
# geometry (sumolib, computed once)
# --------------------------------------------------------------------------
def _via_length(net, a, b):
    conns = net.getEdge(a).getConnections(net.getEdge(b))
    lens = [net.getLane(c.getViaLaneID()).getLength() for c in conns if c.getViaLaneID()]
    return min(lens) if lens else 0.0


def segment_free_flow(net, stops_by_id):
    """{direction: [free-flow time of each stop-to-stop segment]} = route distance / SPEED.
    Distance runs between stop endPos values (a stopped bus has its front at endPos) and
    includes the internal junction lanes."""
    out = {}
    for d, spec in LINES.items():
        edges = spec["edges"]
        ffs = []
        for a, b in zip(spec["stops"], spec["stops"][1:]):
            sa, sb = stops_by_id[a], stops_by_id[b]
            ia, ib = edges.index(sa["edge"]), edges.index(sb["edge"])
            dist = net.getEdge(edges[ia]).getLength() - sa["endPos"]
            for m in range(ia, ib):
                dist += _via_length(net, edges[m], edges[m + 1])
                dist += net.getEdge(edges[m + 1]).getLength() if m + 1 < ib else 0.0
            dist += sb["endPos"]
            ffs.append(dist / SPEED)
        out[d] = ffs
    return out


def approaches(net):
    """{approach id: (tls id, [upstream piece, turn-lane piece])} for all signal approaches."""
    out = {}
    for tid in TLS_IDS:
        for e in {lane.getEdge().getID() for lane, _, _ in net.getTLS(tid).getConnections()}:
            main = e.replace(".250.00", "")
            out[main] = (tid, [main, e])
    return out


def stop_queue_targets(net, stops_by_id):
    """For stops with a signal downstream on the same link: (lane chain from the stop line
    upstream, distance from the stop line to the stop's endPos)."""
    out = {}
    for sid, s in stops_by_id.items():
        piece = s["edge"] + ".250.00"
        if not net.hasEdge(piece):      # exit link: no signal downstream
            continue
        lp = net.getEdge(piece).getLane(0).getLength()
        lm = net.getEdge(s["edge"]).getLane(0).getLength()
        out[sid] = ([(f"{piece}_0", lp, 0.0), (f"{s['edge']}_0", lm, lp)], lp + lm - s["endPos"])
    return out


def queue_tail(sim, lane_chain):
    """Length of the contiguous queue from the stop line over a chain of lanes
    [(lane_id, lane_length, offset_from_stop_line), ...] (downstream lane first)."""
    vehs = []
    for lane_id, length, offset in lane_chain:
        for v in sim.lane.getLastStepVehicleIDs(lane_id):
            vehs.append((offset + length - sim.vehicle.getLanePosition(v), v))
    vehs.sort()
    tail = last = 0.0
    for dist, v in vehs:
        if sim.vehicle.getSpeed(v) >= QUEUE_SPEED or dist - last > QUEUE_GAP:
            break
        last = dist
        tail = dist + sim.vehicle.getLength(v)
    return tail


def spillback(sim, edge_id):
    """True if a halting vehicle sits within SPILL_POS m of the edge's upstream end."""
    return any(sim.vehicle.getSpeed(v) < 0.1 and sim.vehicle.getLanePosition(v) < SPILL_POS
               for v in sim.edge.getLastStepVehicleIDs(edge_id))


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
def run_seed(seed, args, geo):
    t0, t1 = args.warmup, args.warmup + args.duration
    raw = os.path.join(RESULTS, "raw", args.tag, f"seed_{seed}")
    os.makedirs(raw, exist_ok=True)
    routes = os.path.join(HERE, "routes", args.tag, f"routes_seed{seed}.rou.xml")
    gen.generate(seed, routes, t_end=t1, **gen.gen_kwargs(args))
    tripinfo = os.path.join(raw, "tripinfo.xml")
    stopinfo = os.path.join(raw, "stopinfo.xml")

    sim, backend = import_traci(args.gui)
    cmd = [sumolib.checkBinary("sumo-gui" if args.gui else "sumo"),
           "-n", NET, "-r", routes, "-a", STOPS_ADD,
           "--seed", str(seed), "--begin", "0", "--end", str(t1),
           "--tripinfo-output", tripinfo, "--tripinfo-output.write-unfinished", "true",
           "--stop-output", stopinfo,
           "--time-to-teleport", "300", "--no-step-log", "true", "--no-warnings", "true"]
    if args.gui:
        cmd += ["--start", "--quit-on-end"]
    sim.start(cmd)
    controllers = {tid: FixedTimeController(tid) for tid in TLS_IDS}
    for c in controllers.values():
        c.reset(sim)

    rng = np.random.default_rng([seed, 7])          # dwell RNG, independent of demand RNG
    lam = (MEAN_DWELL - T_FIXED) / (T_BOARD * args.headway)
    last_arrival, release, dwells = {}, {}, []

    appr = geo["approaches"]
    max_q = {a: 0 for a in appr}
    spill = {a: [] for a in appr}
    reach = {s: [] for s in geo["stop_queue"]}
    teleports = 0

    t = sim.simulation.getTime()
    while t < t1:
        sim.simulationStep()
        t = sim.simulation.getTime()

        # ---- closed-loop dwell -----------------------------------------
        for v in sim.simulation.getStopStartingVehiclesIDList():
            if not v.startswith(("L1_", "L2_")):
                continue
            stop_id = sim.vehicle.getStops(v, 1)[0].stoppingPlaceID
            gap = t - last_arrival.get(stop_id, t - args.headway)
            last_arrival[stop_id] = t
            dwell, b = dwell_time(rng, gap, lam)
            # the stop began one step before it is reported -> release relative to that
            release[v] = t - 1 + dwell
            dwells.append({"veh": v, "stop": stop_id, "t": t, "gap": gap, "boardings": b,
                           "dwell": dwell})
        for v in [v for v, tr in release.items() if t >= tr]:
            sim.vehicle.resume(v)
            del release[v]

        for c in controllers.values():
            c.act(t, sim)
        teleports += sim.simulation.getStartingTeleportNumber()

        # ---- sampling ----------------------------------------------------
        if t > t0 and int(t) % SAMPLE_DT == 0:
            for a, (_, edges) in appr.items():
                max_q[a] = max(max_q[a], sum(sim.edge.getLastStepHaltingNumber(e) for e in edges))
                spill[a].append(spillback(sim, edges[0]))
            for s, (chain, need) in geo["stop_queue"].items():
                reach[s].append(queue_tail(sim, chain) >= need)
    pending = len(sim.simulation.getPendingVehicles())
    sim.close()

    # ---- metrics -------------------------------------------------------------
    trips = bm.parse_tripinfo(tripinfo)
    stops = bm.parse_stopinfo(stopinfo)
    bus_params = bm.parse_bus_params(routes)
    bus_occ = {v: p["occupancy"] for v, p in bus_params.items()}
    bus_ids = set(bus_params)
    H = args.headway
    car_trips = [tr for tr in bm.in_window(trips, t0, t1) if tr["id"] not in bus_ids]

    row = {"seed": seed, "backend": backend}
    line_rows, all_seg = [], []
    for d, spec in LINES.items():
        prof = bm.line_headway_profile(stops, spec["stops"], H, t0, t1)
        segs = bm.segment_delays(stops, spec["stops"], geo["ff"][d], t0, t1)
        tt = bm.travel_times(trips, {v for v, p in bus_params.items() if p["line"] == d}, t0, t1)
        for k, p in enumerate(prof):
            seg = segs[k] if k < len(segs) else {}
            line_rows.append({
                "seed": seed, "line": d, "stop_idx": k, "stop": p["stop"],
                "n_headways": p["n_headways"], "cv": p["cv"], "bunching": p["bunching"],
                "junction_next": spec["junctions"][k] if k < len(segs) else "",
                "n_seg": len(seg), "seg_delay": bm.mean_or_nan(list(seg.values())),
                "queue_reach_share": (sum(reach[p["stop"]]) / len(reach[p["stop"]])
                                      if p["stop"] in reach and reach[p["stop"]] else math.nan),
            })
        for seg in segs:
            all_seg += list(seg.values())
        cvs = [p["cv"] for p in prof]
        row[f"buses_{d}"] = len(tt)
        row[f"tt_{d}"] = bm.mean_or_nan(list(tt.values()))
        row[f"cv_first_{d}"] = cvs[0]
        row[f"cv_last_{d}"] = cvs[-1]
        row[f"cv_slope_{d}"] = bm.cv_trend(cvs)
        row[f"bunch_{d}"] = sum(p["bunching"] for p in prof)
        row[f"bunch_last_{d}"] = prof[-1]["bunching"]
    row["bunch_total"] = sum(row[f"bunch_{d}"] for d in LINES)
    row["bus_jdelay"] = bm.mean_or_nan(all_seg)
    row["bus_jdelay_gt_cycle"] = sum(1 for x in all_seg if x > args.cycle)
    row["mean_dwell"] = bm.mean_or_nan([x["dwell"] for x in dwells if t0 <= x["t"] < t1])
    row["cars"] = len(car_trips)
    row["car_timeloss"] = bm.mean_or_nan([tr["timeLoss"] for tr in car_trips])
    row["car_depart_delay"] = bm.mean_or_nan([tr["departDelay"] for tr in car_trips])
    row["car_delay_total"] = bm.car_mean_total_delay(trips, bus_ids, t0, t1)
    row["person_delay_h"] = bm.person_delay(trips, bus_occ, gen.CAR_OCCUPANCY, t0, t1,
                                            fields=("timeLoss", "departDelay")) / 3600.0
    row["max_queue_any"] = max(max_q.values())
    row["max_queue_mean"] = float(np.mean(list(max_q.values())))
    longest = {a: bm.longest_run(f, SAMPLE_DT) for a, f in spill.items()}
    row["spill_longest_s"] = max(longest.values())
    row["spill_worst_appr"] = max(longest, key=longest.get)
    row["spill_share_max"] = max(sum(f) / len(f) for f in spill.values())
    shares = [sum(r) / len(r) for r in reach.values() if r]
    row["stopq_reach_mean"] = float(np.mean(shares))
    row["stopq_reach_max"] = float(np.max(shares))
    row["unfinished"] = sum(1 for tr in trips if tr["unfinished"])
    row["never_inserted"] = pending
    row["teleports"] = teleports

    queue_rows = [{"seed": seed, "approach": a, "tls": appr[a][0], "max_halting": max_q[a],
                   "spill_longest_s": longest[a], "spill_share": sum(spill[a]) / len(spill[a])}
                  for a in appr]
    pd.DataFrame(dwells).to_csv(os.path.join(raw, "dwells.csv"), index=False)
    return row, line_rows, queue_rows


def main():
    ap = gen.add_args(argparse.ArgumentParser(description="BO4 grid fixed-time baseline"))
    # team rule: evaluation seeds start at 10000
    ap.add_argument("--seeds", type=int, nargs="+", default=[10000, 10001, 10002, 10003, 10004])
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--warmup", type=float, default=600.0)
    ap.add_argument("--duration", type=float, default=7200.0)
    ap.add_argument("--tag", default="grid_fixed_time",
                    help="run name: results/<tag>*.csv, raw outputs in results/raw/<tag>")
    args = ap.parse_args()

    net = sumolib.net.readNet(NET, withInternal=True)
    stops_by_id = {s.id: {"edge": s.lane.rsplit("_", 1)[0], "endPos": float(s.endPos)}
                   for s in sumolib.xml.parse(STOPS_ADD, "busStop")}
    args.cycle = sum(float(p.duration) for p in sumolib.xml.parse(NET, "phase")) / len(TLS_IDS)
    geo = {"ff": segment_free_flow(net, stops_by_id), "approaches": approaches(net),
           "stop_queue": stop_queue_targets(net, stops_by_id)}
    for d, ffs in geo["ff"].items():
        print(f"{d}: segment free-flow times {[round(x, 1) for x in ffs]} s "
              f"(route distance / {SPEED} m/s)")

    rows, lines, queues = [], [], []

    for seed in args.seeds:
        tic = time.time()
        r, l, q = run_seed(seed, args, geo)
        r["wall_s"] = round(time.time() - tic, 1)
        print(f"--- seed {seed} done in {r['wall_s']} s", flush=True)
        rows.append(r)
        lines += l
        queues += q

    os.makedirs(RESULTS, exist_ok=True)
    df = pd.DataFrame(rows)
    dl = pd.DataFrame(lines)
    dq = pd.DataFrame(queues)
    df.to_csv(os.path.join(RESULTS, f"{args.tag}.csv"), index=False)
    dl.to_csv(os.path.join(RESULTS, f"{args.tag}_lines.csv"), index=False)
    dq.to_csv(os.path.join(RESULTS, f"{args.tag}_queues.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 100)
    show = df.drop(columns=["backend", "spill_worst_appr"]).set_index("seed")
    print("\nPer seed:")
    print(show.round(2).T.to_string())
    print("\nWorst spillback approach per seed:", dict(zip(df.seed, df.spill_worst_appr)))
    print("\nPer line / stop (mean over seeds):")
    print(dl.groupby(["line", "stop_idx", "stop", "junction_next"], sort=False)
            [["n_headways", "cv", "bunching", "n_seg", "seg_delay", "queue_reach_share"]]
            .mean().round(2).to_string())
    print("\nMean +- std over seeds:")
    for c in show.columns:
        print(f"  {c:<22} {show[c].mean():9.2f} +- {show[c].std():.2f}")
    print("\nsaved", os.path.join(RESULTS, f"{args.tag}*.csv"))


if __name__ == "__main__":
    main()
