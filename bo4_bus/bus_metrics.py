"""
Bus / car metrics for the BO4 scenario.

Parsing helpers turn SUMO XML outputs into plain lists of dicts; every metric
is a pure function over those lists (no SUMO needed -> unit-testable).

Measurement window: only events in [t0, t1) are counted (t0 = end of warm-up).
"""
import math
import statistics
import xml.etree.ElementTree as ET


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_tripinfo(path):
    """List of dicts: id, vType, depart, arrival (-1 if unfinished), timeLoss, departDelay,
    unfinished."""
    trips = []
    for el in ET.parse(path).getroot().iter("tripinfo"):
        arrival = float(el.get("arrival", -1))
        trips.append({
            "id": el.get("id"),
            "vType": el.get("vType"),
            "depart": float(el.get("depart", -1)),
            "arrival": arrival,
            "timeLoss": float(el.get("timeLoss", 0.0)),
            "departDelay": float(el.get("departDelay", 0.0)),
            "unfinished": arrival < 0 or el.get("vaporized", "") not in ("", None),
        })
    return trips


def parse_stopinfo(path):
    """List of dicts: id, busStop, started, ended."""
    stops = []
    for el in ET.parse(path).getroot().iter("stopinfo"):
        stops.append({
            "id": el.get("id"),
            "busStop": el.get("busStop"),
            "started": float(el.get("started")),
            "ended": float(el.get("ended", -1)),
        })
    return stops


def parse_bus_params(route_path):
    """{bus_id: {"line": str, "occupancy": float}} from <param> children of the route file."""
    out = {}
    for veh in ET.parse(route_path).getroot().iter("vehicle"):
        params = {p.get("key"): p.get("value") for p in veh.findall("param")}
        if "line" in params:
            out[veh.get("id")] = {"line": params["line"],
                                  "occupancy": float(params.get("occupancy", 0))}
    return out


def parse_edgedata(path):
    """{edge_id: {"timeLoss": total s, "left": vehicles}} from a single-interval
    edgeData output (edges without traffic are absent)."""
    out = {}
    for el in ET.parse(path).getroot().iter("edge"):
        out[el.get("id")] = {"timeLoss": float(el.get("timeLoss", 0.0)),
                             "left": int(float(el.get("left", 0)))}
    return out


def approach_delay(edgedata, approach_edges):
    """Mean delay (s) per approach passage over a set of signal approaches.
    approach_edges: {approach: [upstream piece, ..., piece ending at the stop line]};
    delay = sum of time loss on all pieces / vehicles that left the last piece."""
    loss = sum(edgedata.get(e, {}).get("timeLoss", 0.0)
               for edges in approach_edges.values() for e in edges)
    passed = sum(edgedata.get(edges[-1], {}).get("left", 0) for edges in approach_edges.values())
    return loss / passed if passed else math.nan


# --------------------------------------------------------------------------
# generic statistics
# --------------------------------------------------------------------------
def headways(times):
    """Successive differences of the sorted event times."""
    ts = sorted(times)
    return [b - a for a, b in zip(ts, ts[1:])]


def cv(values):
    """Coefficient of variation = sample std (ddof=1) / mean. nan if < 2 values."""
    if len(values) < 2:
        return math.nan
    m = statistics.fmean(values)
    return statistics.stdev(values) / m if m > 0 else math.nan


def count_bunching(hws, target_headway, factor=0.5):
    """Number of headways strictly below factor * target_headway."""
    return sum(1 for h in hws if h < factor * target_headway)


def mean_or_nan(values):
    return statistics.fmean(values) if values else math.nan


# --------------------------------------------------------------------------
# bus metrics
# --------------------------------------------------------------------------
def free_flow_time(distance_m, speed_mps):
    """Free-flow travel time = distance / speed (no acceleration/deceleration)."""
    return distance_m / speed_mps


def stop_times(stops, bus_stop, t0, t1, field="started"):
    """{veh_id: time} for one bus stop, keeping only times in [t0, t1)."""
    return {s["id"]: s[field] for s in stops
            if s["busStop"] == bus_stop and t0 <= s[field] < t1}


def bus_junction_delays(stops, up_stop, down_stop, ff_time, t0, t1):
    """
    {veh_id: delay} where delay = (started at down_stop - ended at up_stop) - ff_time.
    A bus is counted if it left the upstream stop at t >= t0 and reached the
    downstream stop before t1.
    """
    left_up = {s["id"]: s["ended"] for s in stops
               if s["busStop"] == up_stop and s["ended"] >= t0}
    arr_down = {s["id"]: s["started"] for s in stops
                if s["busStop"] == down_stop and s["started"] < t1}
    return {v: arr_down[v] - left_up[v] - ff_time
            for v in left_up if v in arr_down and arr_down[v] >= left_up[v]}


def headway_stats(stops, bus_stop, target_headway, t0, t1, factor=0.5):
    """Headways (from `started`) at one stop within [t0, t1): list, CV, bunching count."""
    hws = headways(stop_times(stops, bus_stop, t0, t1).values())
    return {"headways": hws, "cv": cv(hws),
            "bunching": count_bunching(hws, target_headway, factor)}


# --------------------------------------------------------------------------
# car / person metrics
# --------------------------------------------------------------------------
def in_window(trips, t0, t1):
    """Trips that departed in [t0, t1)."""
    return [t for t in trips if t0 <= t["depart"] < t1]


def car_mean_delay(trips, bus_ids, t0, t1, field="timeLoss"):
    """Mean `field` (timeLoss by default) of non-bus trips that departed in [t0, t1)
    (unfinished included)."""
    return mean_or_nan([t[field] for t in in_window(trips, t0, t1)
                        if t["id"] not in bus_ids])


def person_delay(trips, bus_occupancy, car_occupancy, t0, t1, fields=("timeLoss",)):
    """sum(delay * occupancy) over trips departed in [t0, t1); buses use their own occupancy.
    delay = sum of `fields` per trip (timeLoss by default; add "departDelay" to include
    insertion delay)."""
    total = 0.0
    for t in in_window(trips, t0, t1):
        occ = bus_occupancy.get(t["id"], car_occupancy)
        total += sum(t[f] for f in fields) * occ
    return total


# --------------------------------------------------------------------------
# multi-stop lines (grid scenario)
# --------------------------------------------------------------------------
def total_delay(trip):
    """timeLoss + departDelay of one trip (driving delay + insertion delay)."""
    return trip["timeLoss"] + trip["departDelay"]


def car_mean_total_delay(trips, bus_ids, t0, t1):
    """Mean timeLoss + departDelay of non-bus trips departed in [t0, t1) (unfinished included)."""
    return mean_or_nan([total_delay(t) for t in in_window(trips, t0, t1)
                        if t["id"] not in bus_ids])


def line_headway_profile(stops, stop_ids, target_headway, t0, t1, factor=0.5):
    """Headway CV and bunching count at each stop of a line, in line order.
    Returns [{"stop", "n_headways", "cv", "bunching"}, ...]."""
    out = []
    for sid in stop_ids:
        hs = headway_stats(stops, sid, target_headway, t0, t1, factor)
        out.append({"stop": sid, "n_headways": len(hs["headways"]),
                    "cv": hs["cv"], "bunching": hs["bunching"]})
    return out


def cv_trend(cvs):
    """Least-squares slope of CV vs stop index (CV units per stop); nan if < 2 finite values."""
    pts = [(i, c) for i, c in enumerate(cvs) if not math.isnan(c)]
    if len(pts) < 2:
        return math.nan
    xs, ys = zip(*pts)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in pts) / sxx


def segment_delays(stops, stop_ids, ff_times, t0, t1):
    """Delay on each stop-to-stop segment of a line (each segment crosses one junction):
    list (len(stop_ids) - 1) of {veh_id: delay}, see bus_junction_delays()."""
    return [bus_junction_delays(stops, a, b, ff, t0, t1)
            for a, b, ff in zip(stop_ids, stop_ids[1:], ff_times)]


def travel_times(trips, veh_ids, t0, t1):
    """{veh_id: arrival - depart} for finished trips of veh_ids departed in [t0, t1)
    and arrived before t1."""
    return {t["id"]: t["arrival"] - t["depart"] for t in in_window(trips, t0, t1)
            if t["id"] in veh_ids and not t["unfinished"] and t["arrival"] < t1}


def longest_run(flags, dt=1.0):
    """Longest run of consecutive True values in a sampled boolean series, in time units."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best * dt
