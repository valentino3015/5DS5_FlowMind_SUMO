"""
BO4 bus scenario - fixed-time baseline.

    python bo4_bus/Bus_Demo.py --seeds 1 2 3 4 5
    python bo4_bus/Bus_Demo.py --seeds 1 --gui

For each seed: generate routes (gen_routes.py), run SUMO headless for
warm-up (600 s) + measurement (3600 s), then compute the metrics of
bus_metrics.py over the measurement window only. Prints a per-seed table and
mean +- std, and writes bo4_bus/results/fixed_time.csv.

The signal controller is a plug-in object with `act(t, traci)` called after
every simulation step; FixedTimeController does nothing (the static program
stored in the net runs on its own).
"""
import argparse
import math
import os
import sys

if "SUMO_HOME" not in os.environ:
    for _p in (r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Program Files\Eclipse\Sumo"):
        if os.path.isdir(_p):
            os.environ["SUMO_HOME"] = _p
            break

import pandas as pd
import sumolib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bus_metrics as bm  # noqa: E402
import gen_routes  # noqa: E402

NET = os.path.join(HERE, "nets", "bus.net.xml")
STOPS_ADD = os.path.join(HERE, "nets", "bus_stops.add.xml")
RESULTS = os.path.join(HERE, "results")

DIRS = {"EB": ("EB_up", "EB_down"), "WB": ("WB_up", "WB_down")}
ART_APPROACHES = ("W2C", "E2C")
CROSS_APPROACHES = ("N2C", "S2C")
CYCLE = 71.0           # s, fixed-time cycle of bus.net.xml
QUEUE_GAP = 25.0       # m, max gap between consecutive queued vehicles of one queue
QUEUE_SPEED = 1.39     # m/s (5 km/h): slower vehicles count as queued (stop-and-go)


class FixedTimeController:
    """Baseline: leaves the static program in the net untouched."""

    def reset(self, traci):
        pass

    def act(self, t, traci):
        pass


def import_traci(gui):
    if not gui:
        try:
            import libsumo as traci
            return traci, "libsumo"
        except ImportError:
            pass
    import traci
    return traci, "traci"


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def stop_to_stop_distances():
    """Distance (m) from upstream-stop endPos to downstream-stop endPos, per direction.
    A stopped bus has its front at endPos, so this is the distance its front travels."""
    net = sumolib.net.readNet(NET, withInternal=True)
    stops = {s.id: s for s in sumolib.xml.parse(STOPS_ADD, "busStop")}
    out = {}
    for d, (up, down) in DIRS.items():
        s_up, s_down = stops[up], stops[down]
        lane_up = net.getLane(s_up.lane)
        conn = [c for c in lane_up.getOutgoing() if c.getToLane().getID() == s_down.lane][0]
        # straight movement: a single internal lane (no internal junction)
        internal = net.getLane(conn.getViaLaneID()).getLength()
        out[d] = (lane_up.getLength() - float(s_up.endPos)) + internal + float(s_down.endPos)
    return out


def queue_length(traci, lane_id, lane_len):
    """Length (m from the stop line) of the contiguous chain of queued vehicles
    (speed < QUEUE_SPEED, consecutive gaps <= QUEUE_GAP)."""
    vehs = [(lane_len - traci.vehicle.getLanePosition(v), v)
            for v in traci.lane.getLastStepVehicleIDs(lane_id)]
    vehs.sort()
    tail, last = 0.0, 0.0
    for dist, v in vehs:
        if traci.vehicle.getSpeed(v) >= QUEUE_SPEED or dist - last > QUEUE_GAP:
            break
        last = dist
        tail = dist + traci.vehicle.getLength(v)
    return tail


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
def run_seed(seed, args, controller, ff_times):
    t0, t1 = args.warmup, args.warmup + args.duration
    raw = os.path.join(RESULTS, "raw", args.tag, f"seed_{seed}")
    os.makedirs(raw, exist_ok=True)
    routes = os.path.join(HERE, "routes", args.tag, f"routes_seed{seed}.rou.xml")
    gen_routes.generate(seed, routes, t_end=t1, **gen_routes.gen_kwargs(args))
    tripinfo = os.path.join(raw, "tripinfo.xml")
    stopinfo = os.path.join(raw, "stopinfo.xml")

    traci, backend = import_traci(args.gui)
    cmd = [sumolib.checkBinary("sumo-gui" if args.gui else "sumo"),
           "-n", NET, "-r", routes, "-a", STOPS_ADD,
           "--seed", str(seed), "--begin", "0", "--end", str(t1),
           "--tripinfo-output", tripinfo, "--tripinfo-output.write-unfinished", "true",
           "--stop-output", stopinfo,
           "--time-to-teleport", "300", "--no-step-log", "true", "--no-warnings", "true"]
    if args.gui:
        cmd += ["--start", "--quit-on-end"]
    traci.start(cmd)
    controller.reset(traci)

    art_lanes = {lid: traci.lane.getLength(lid)
                 for e in ART_APPROACHES for lid in (f"{e}_0", f"{e}_1")}
    max_cross_q = {e: 0 for e in CROSS_APPROACHES}
    art_q = {e: [] for e in ART_APPROACHES}   # per-second queue length (m)
    teleports = 0
    t = traci.simulation.getTime()
    while t < t1:
        traci.simulationStep()
        t = traci.simulation.getTime()
        controller.act(t, traci)
        teleports += traci.simulation.getStartingTeleportNumber() if t > t0 else 0
        if t > t0:
            for e in CROSS_APPROACHES:
                max_cross_q[e] = max(max_cross_q[e], traci.edge.getLastStepHaltingNumber(e))
            for e in ART_APPROACHES:
                art_q[e].append(max(queue_length(traci, f"{e}_{i}", art_lanes[f"{e}_{i}"])
                                    for i in (0, 1)))
    pending = len(traci.simulation.getPendingVehicles())
    traci.close()

    # ---- metrics ---------------------------------------------------------
    trips = bm.parse_tripinfo(tripinfo)
    stops = bm.parse_stopinfo(stopinfo)
    bus_params = bm.parse_bus_params(routes)
    bus_occ = {v: p["occupancy"] for v, p in bus_params.items()}
    H = args.headway

    row = {"seed": seed, "backend": backend}
    all_delays = []
    for d, (up, down) in DIRS.items():
        delays = bm.bus_junction_delays(stops, up, down, ff_times[d], t0, t1)
        all_delays += list(delays.values())
        hs_up = bm.headway_stats(stops, up, H, t0, t1)
        hs_dn = bm.headway_stats(stops, down, H, t0, t1)
        row[f"buses_{d}"] = len(delays)
        row[f"bus_delay_{d}"] = bm.mean_or_nan(list(delays.values()))
        row[f"cv_up_{d}"] = hs_up["cv"]
        row[f"cv_down_{d}"] = hs_dn["cv"]
        row[f"bunch_up_{d}"] = hs_up["bunching"]
        row[f"bunch_down_{d}"] = hs_dn["bunching"]
    row["bus_delay"] = bm.mean_or_nan(all_delays)
    row["bus_delay_max"] = max(all_delays) if all_delays else math.nan
    row["bus_wait_gt_cycle"] = sum(1 for x in all_delays if x > CYCLE)
    row["car_delay"] = bm.car_mean_delay(trips, set(bus_params), t0, t1)
    row["car_depart_delay"] = bm.car_mean_delay(trips, set(bus_params), t0, t1,
                                                field="departDelay")
    row["person_delay_h"] = bm.person_delay(trips, bus_occ, gen_routes.CAR_OCCUPANCY,
                                            t0, t1) / 3600.0
    for e in CROSS_APPROACHES:
        row[f"max_q_{e}"] = max_cross_q[e]
    for e in ART_APPROACHES:
        q = art_q[e]
        row[f"art_qmax_m_{e}"] = max(q)
        # share of seconds the queue tail reaches the downstream end of the upstream stop
        row[f"art_q_at_stop_{e}"] = sum(1 for x in q if x >= 135.0) / len(q)
    row["unfinished"] = sum(1 for tr in trips if tr["unfinished"])
    row["pending_insert"] = pending
    row["teleports"] = teleports
    return row


def main():
    ap = gen_routes.add_args(argparse.ArgumentParser(description="BO4 fixed-time bus baseline"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--warmup", type=float, default=600.0)
    ap.add_argument("--duration", type=float, default=3600.0)
    ap.add_argument("--tag", default="fixed_time",
                    help="run name: raw outputs go to results/raw/<tag>, routes to routes/<tag>")
    ap.add_argument("--out", default=None, help="CSV path (default results/<tag>.csv)")
    args = ap.parse_args()
    args.out = args.out or os.path.join(RESULTS, f"{args.tag}.csv")

    dist = stop_to_stop_distances()
    speed = 13.89  # m/s: lane speed limit = bus maxSpeed (speedFactor 1)
    ff = {d: bm.free_flow_time(x, speed) for d, x in dist.items()}
    for d in DIRS:
        print(f"{d}: stop-to-stop distance {dist[d]:.1f} m -> free-flow {ff[d]:.1f} s "
              f"(distance / {speed} m/s)")

    rows = []
    for seed in args.seeds:
        print(f"--- seed {seed} ...", flush=True)
        rows.append(run_seed(seed, args, FixedTimeController(), ff))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 100)
    show = df.drop(columns=["backend"]).set_index("seed")
    print("\nPer seed (delays in s, queues in veh or m):")
    print(show.round(2).T.to_string())
    print("\nMean +- std over seeds:")
    for c in show.columns:
        print(f"  {c:<22} {show[c].mean():9.2f} +- {show[c].std():.2f}")
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
