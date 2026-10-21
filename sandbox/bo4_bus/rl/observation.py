"""
observe(state) -> float32 vector, pure function of a JunctionState (no SUMO).

Layout (fixed length OBS_SIZE = 5 * len(APPROACHES) + len(green_phases) + 2 = 24 for B1):
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
"""
import numpy as np

from sense import APPROACHES, approach_of

H = 180.0             # s, scheduled headway
DETECT = 150.0        # m, bus range used in the observation
OCC_MAX = 60.0        # persons, max bus occupancy in the demand
MAX_GREEN = 60.0      # s, team timing
JAM_SPACING = 7.5     # m per queued car (5 m length + 2.5 m min gap)
N_GREENS = 2
OBS_SIZE = 5 * len(APPROACHES) + N_GREENS + 2

LOW = np.array(([0.0, 0.0, 0.0, -1.0, 0.0] * len(APPROACHES)) + [0.0] * (N_GREENS + 2),
               dtype=np.float32)
HIGH = np.ones(OBS_SIZE, dtype=np.float32)


def _clip(x, lo, hi):
    return min(max(x, lo), hi)


def observe(state, approaches=APPROACHES):
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    halting = dict.fromkeys(approaches, 0)
    capacity = dict.fromkeys(approaches, 0.0)
    for lane in state.incoming:
        a = approach_of(lane.lane_id)
        if a in halting:
            halting[a] += lane.halting_count
            capacity[a] += lane.length / JAM_SPACING
    nearest = {}
    for bus in state.approaching:
        a = approach_of(bus.lane_id)
        if bus.distance_to_stopline <= DETECT and (
                a not in nearest or bus.distance_to_stopline < nearest[a].distance_to_stopline):
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
    o = 5 * len(approaches)
    obs[o + state.current_green] = 1.0
    obs[o + N_GREENS] = 1.0 if state.in_transition else 0.0
    obs[o + N_GREENS + 1] = _clip(state.time_in_phase / MAX_GREEN, 0.0, 1.0)
    return obs
