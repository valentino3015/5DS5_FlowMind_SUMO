"""
observe(state, version="v1") -> float32 vector, pure function of a JunctionState (no SUMO).

v1 (default; the trained RL-D and RL-E models use it), fixed length
OBS_SIZE = 5 * len(APPROACHES) + len(green_phases) + 2 = 24 for B1:
    per incoming approach, in APPROACHES order:
        queue        halting vehicles / approach capacity (sum of lane lengths / 7.5 m), [0, 1]
        bus_present  1 if a bus is within 150 m of the stop line, else 0
        bus_dist     distance of the nearest such bus / 150 m, [0, 1]      (0 if no bus)
        bus_late     its headway lateness / H, clipped to [-1, 1]          (0 if no bus or unknown)
        bus_occ      its persons on board / 60, [0, 1]                      (0 if no bus)
    current green phase one-hot (one entry per green phase)
    in_transition    1 during yellow / all-red
    time_in_phase    s since the current green started / max green, [0, 1]
bus_present is added because bus_dist = 0 also means "bus at the stop line".

v2 = the 24 values of v1 in the same order, then per incoming approach (APPROACHES order),
OBS_SIZE_V2 = 24 + 3 * 4 = 36:
        since_prev   time since the previous bus of the nearest bus's line passed B1 / H,
                     clipped to [0, 2] (= lateness / H + 1; 0 if no bus or unknown).
                     Unlike bus_late it does not saturate at 1 H late.
        second_bus   1 if a second bus is within 150 m on that approach (a bunched pair)
        out_occ      max occupancy of the outgoing lanes straight ahead of that approach
                     (spillback risk: a green there is wasted if the exit is full), [0, 1]
"""
import numpy as np

from sense import APPROACHES, approach_of

H = 180.0             # s, scheduled headway
DETECT = 150.0        # m, bus range used in the observation
OCC_MAX = 60.0        # persons, max bus occupancy in the demand
MAX_GREEN = 60.0      # s, team timing
JAM_SPACING = 7.5     # m per queued car (5 m length + 2.5 m min gap)
SINCE_MAX = 2.0       # since_prev is clipped to [0, 2] headways
N_GREENS = 2
VERSIONS = ("v1", "v2")
OBS_SIZE = 5 * len(APPROACHES) + N_GREENS + 2
OBS_SIZE_V2 = OBS_SIZE + 3 * len(APPROACHES)

LOW = np.array(([0.0, 0.0, 0.0, -1.0, 0.0] * len(APPROACHES)) + [0.0] * (N_GREENS + 2),
               dtype=np.float32)
HIGH = np.ones(OBS_SIZE, dtype=np.float32)
LOW_V2 = np.concatenate([LOW, np.zeros(3 * len(APPROACHES), dtype=np.float32)])
HIGH_V2 = np.concatenate([HIGH, np.array([SINCE_MAX, 1.0, 1.0] * len(APPROACHES),
                                         dtype=np.float32)])


def bounds(version="v1"):
    """(low, high) of the observation vector of `version`."""
    return {"v1": (LOW, HIGH), "v2": (LOW_V2, HIGH_V2)}[version]


def version_of_size(n):
    """Observation version from a vector length (e.g. a trained model's input size)."""
    return {OBS_SIZE: "v1", OBS_SIZE_V2: "v2"}[n]


def straight_out(approach):
    """Outgoing edge straight ahead of an incoming approach of the grid, 'A1B1' -> 'B1C1'
    (node ids are a column letter + a row digit)."""
    (c0, r0), (c1, r1) = approach[:2], approach[2:4]
    return f"{c1}{r1}{chr(2 * ord(c1) - ord(c0))}{2 * int(r1) - int(r0)}"


def _clip(x, lo, hi):
    return min(max(x, lo), hi)


def observe(state, approaches=APPROACHES, version="v1"):
    if version not in VERSIONS:
        raise ValueError(f"unknown observation version {version!r}")
    n = len(approaches)
    obs = np.zeros(OBS_SIZE if version == "v1" else OBS_SIZE_V2, dtype=np.float32)
    halting = dict.fromkeys(approaches, 0)
    capacity = dict.fromkeys(approaches, 0.0)
    for lane in state.incoming:
        a = approach_of(lane.lane_id)
        if a in halting:
            halting[a] += lane.halting_count
            capacity[a] += lane.length / JAM_SPACING
    nearest, n_buses = {}, dict.fromkeys(approaches, 0)
    for bus in state.approaching:
        a = approach_of(bus.lane_id)
        if bus.distance_to_stopline <= DETECT:
            if a in n_buses:
                n_buses[a] += 1
            if a not in nearest or bus.distance_to_stopline < nearest[a].distance_to_stopline:
                nearest[a] = bus
    for k, a in enumerate(approaches):
        o = 5 * k
        obs[o] = _clip(halting[a] / capacity[a], 0.0, 1.0) if capacity[a] else 0.0
        bus = nearest.get(a)
        if bus is not None:
            obs[o + 1] = 1.0
            obs[o + 2] = _clip(bus.distance_to_stopline / DETECT, 0.0, 1.0)
            if bus.headway_lateness is not None:
                obs[o + 3] = _clip(bus.headway_lateness / H, -1.0, 1.0)
            obs[o + 4] = _clip((bus.persons_on_board or 0) / OCC_MAX, 0.0, 1.0)
    o = 5 * n
    obs[o + state.current_green] = 1.0
    obs[o + N_GREENS] = 1.0 if state.in_transition else 0.0
    obs[o + N_GREENS + 1] = _clip(state.time_in_phase / MAX_GREEN, 0.0, 1.0)
    if version == "v1":
        return obs

    out_occ = {}
    for lane in state.outgoing:
        e = approach_of(lane.lane_id)
        out_occ[e] = max(out_occ.get(e, 0.0), lane.occupancy)
    o = OBS_SIZE
    for k, a in enumerate(approaches):
        bus = nearest.get(a)
        if bus is not None and bus.headway_lateness is not None:
            obs[o + 3 * k] = _clip((bus.headway_lateness + H) / H, 0.0, SINCE_MAX)
        obs[o + 3 * k + 1] = 1.0 if n_buses[a] >= 2 else 0.0
        obs[o + 3 * k + 2] = _clip(out_occ.get(straight_out(a), 0.0), 0.0, 1.0)
    return obs
