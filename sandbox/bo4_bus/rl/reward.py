"""
Rewards for the last decision tick, pure functions of JunctionStates (no SUMO).
Each returns (reward, terms); terms are unscaled values for logging.

Main rewards (delay incurred this tick, as in the BO4 delay metrics):
D  reward_vehicle(prev, state) = -(halting vehicles on the incoming lanes) * dt
E  reward_person(prev, state)  = -(halting persons on the incoming lanes) * dt,
   cars weighted CAR_OCCUPANCY, buses their persons_on_board.
F  reward_regular(prev, state, w) = E
   - w * sum over halting buses at B1 of (max(0, lateness) / H) * persons_on_board * dt:
   a queued late bus costs its passengers' delay (in E) plus w x its lateness in
   headways, so the agent is pushed to serve late buses (regularity) and not early ones.
   Unknown lateness (no previous passage on record) counts as 0.
G  reward_regular_g(prev, state, w, w_early, c_switch) = F (same w)
   - w_early * [no late bus on any approach] * sum over early buses on the approaches
       that have green now of (-lateness / H) * persons_on_board * dt
   - c_switch * SWITCH_UNIT * [the green changed during the tick]
   Early-bus penalty: C's rule (no priority for an early bus) made soft. Each second of
   green given or held for an approach with an early bus costs that bus's passengers
   x its earliness in headways, the mirror of F's lateness term, but only while no late
   bus is waiting (a late bus always has precedence, as in C). "Bus on an approach" =
   within DETECT = 150 m of the stop line (the observation's range), not dwelling at a
   stop; "late" = lateness > 0, "early" = lateness < 0 (unknown counts as neither).
   Green approaches: green 0 (E-W) serves the W and E slots, green 1 the S and N slots.
   Switch cost: SWITCH_UNIT = 100 person-seconds = 20 persons waiting through the 5 s
   change interval (yellow 3 + all-red 2), the order of one tick of reward E at B1;
   a switch = prev.current_green != state.current_green (a switch fills one tick).
G2 reward_regular_g2(prev, state, events, w_delay, w_headway, c_switch) =
     w_delay * E
   - w_headway * HEADWAY_UNIT * sum over buses that passed the junction this tick of
       ((gap - H) / H)^2                  (gap = time since the previous bus of the same
                                           line passed this junction; skipped if unknown)
   - c_switch * SWITCH_UNIT * [the green changed during the tick]
   The headway term scores regularity where the signal acts on it: an early bus passing
   (small gap) and a late bus passing (large gap) both cost, so holding early buses and
   speeding late ones pays, unlike E/F/G where any bus saved is a gain. `events` are the
   Sensor's passages of the tick (env.py passes them). HEADWAY_UNIT = 1000 person-s:
   with ~80 bus passages per hour at B1 and a mean ((gap-H)/H)^2 of about 0.2, w_headway
   = 4 weighs regularity about as much as one hour of reward E.
"Halting" = below 0.1 m/s (LaneState.halting_count). A bus dwelling at a stop is
halted but is not signal delay, so it is excluded (VehicleInfo.at_stop). Car
halting = lane halting - all halting buses; the sensor covers the whole approach,
so every bus on the incoming lanes is in `approaching`. `prev` is unused (kept for
the LearningSpecialist.reward(prev, state, events) signature).

Reference only (phase 1, not used for training):
reward_vehicle_diff / reward_person_diff = -(change in total accumulated waiting time).
Their episode sum telescopes to (waiting at start - waiting at end), ~0 for any policy.
"""
from sense import APPROACHES, approach_of

CAR_OCCUPANCY = 1.3
TICK = 5.0            # s, decision interval
H = 180.0             # s, scheduled headway
HALT_SPEED = 0.1      # m/s, SUMO's halting threshold
DETECT = 150.0        # m, bus range of the early/late test (= observation.DETECT)
SWITCH_UNIT = 100.0   # person-seconds per switch at c_switch = 1
HEADWAY_UNIT = 1000.0  # person-seconds per (headway deviation / H)^2 at w_headway = 1
GREEN_OF_SLOT = (0, 0, 1, 1)   # approach slots W, E, S, N -> green index serving them


def _incoming_buses(state):
    lanes = {l.lane_id for l in state.incoming}
    return [b for b in state.approaching if b.vclass == "bus" and b.lane_id in lanes]


def halting_counts(state, car_occupancy=CAR_OCCUPANCY):
    """(halting vehicles, halting persons) on the incoming lanes, dwelling buses excluded."""
    total = sum(l.halting_count for l in state.incoming)
    halted = [b for b in _incoming_buses(state) if b.speed < HALT_SPEED]
    queued = [b for b in halted if not b.at_stop]
    cars = max(total - len(halted), 0)
    persons = car_occupancy * cars + sum(b.persons_on_board or 0 for b in queued)
    return cars + len(queued), persons, cars, len(queued)


def _queued_buses(state):
    """Halting buses on the incoming lanes that are not dwelling at a stop."""
    return [b for b in _incoming_buses(state) if b.speed < HALT_SPEED and not b.at_stop]


def reward_vehicle(prev, state, dt=TICK):
    veh, _, cars, buses = halting_counts(state)
    r = -veh * dt
    return r, {"r_vehicle": r, "halting_veh": float(veh), "halting_cars": float(cars),
               "halting_buses": float(buses), "queue": float(state.total_queue())}


def reward_person(prev, state, dt=TICK, car_occupancy=CAR_OCCUPANCY):
    veh, persons, cars, buses = halting_counts(state, car_occupancy)
    bus_persons = persons - car_occupancy * cars
    r = -persons * dt
    return r, {"r_person": r, "halting_persons": persons, "halting_bus_persons": bus_persons,
               "halting_veh": float(veh), "halting_buses": float(buses),
               "queue": float(state.total_queue())}


def reward_regular(prev, state, dt=TICK, w=1.0, headway=H, car_occupancy=CAR_OCCUPANCY):
    r_person, terms = reward_person(prev, state, dt, car_occupancy)
    late = sum(max(0.0, b.headway_lateness or 0.0) / headway * (b.persons_on_board or 0)
               for b in _queued_buses(state))
    r_late = -w * late * dt
    r = r_person + r_late
    return r, {"r_regular": r, **terms, "r_late": r_late, "late_person_headways": late}


def _buses_in_range(state):
    """Buses within DETECT of the stop line, not dwelling at a stop, on the incoming lanes."""
    return [b for b in _incoming_buses(state)
            if b.distance_to_stopline <= DETECT and not b.at_stop]


def reward_regular_g(prev, state, dt=TICK, w=5.0, w_early=1.0, c_switch=1.0, headway=H,
                     car_occupancy=CAR_OCCUPANCY, approaches=APPROACHES):
    r_f, terms = reward_regular(prev, state, dt, w, headway, car_occupancy)
    buses = _buses_in_range(state)
    late_waiting = any((b.headway_lateness or 0.0) > 0 for b in buses)
    green_appr = {a for a, g in zip(approaches, GREEN_OF_SLOT)
                  if a is not None and g == state.current_green}
    early = 0.0
    if not late_waiting:
        early = sum(-b.headway_lateness / headway * (b.persons_on_board or 0) for b in buses
                    if b.headway_lateness is not None and b.headway_lateness < 0
                    and approach_of(b.lane_id) in green_appr)
    r_early = -w_early * early * dt
    switched = prev is not None and prev.current_green != state.current_green
    r_switch = -c_switch * SWITCH_UNIT * float(switched)
    r = r_f + r_early + r_switch
    return r, {"r_regular_g": r, **terms, "r_early": r_early, "early_person_headways": early,
               "late_bus_waiting": float(late_waiting), "r_switch": r_switch,
               "switched": float(switched)}


def reward_regular_g2(prev, state, events=(), dt=TICK, w_delay=1.0, w_headway=4.0,
                      c_switch=0.5, headway=H, car_occupancy=CAR_OCCUPANCY):
    r_person, terms = reward_person(prev, state, dt, car_occupancy)
    devs = [((e["gap"] - headway) / headway) ** 2 for e in events if e.get("gap") is not None]
    dev2 = sum(devs)
    r_headway = -w_headway * HEADWAY_UNIT * dev2
    switched = prev is not None and prev.current_green != state.current_green
    r_switch = -c_switch * SWITCH_UNIT * float(switched)
    r = w_delay * r_person + r_headway + r_switch
    return r, {"r_regular_g2": r, **terms, "r_headway": r_headway, "headway_dev2": dev2,
               "passages": float(len(devs)), "r_switch": r_switch, "switched": float(switched)}


# ---------------------------------------------------------------- reference (phase 1)
def _bus_waiting(state):
    """(unweighted, occupancy-weighted) waiting of buses on the incoming lanes."""
    w = wp = 0.0
    for b in _incoming_buses(state):
        w += b.waiting_time
        wp += b.waiting_time * (b.persons_on_board or 0)
    return w, wp


def vehicle_waiting(state):
    return sum(l.waiting_time for l in state.incoming)


def person_waiting(state, car_occupancy=CAR_OCCUPANCY):
    total = vehicle_waiting(state)
    bus, bus_p = _bus_waiting(state)
    return car_occupancy * max(total - bus, 0.0) + bus_p


def reward_vehicle_diff(prev, state):
    w0, w1 = vehicle_waiting(prev), vehicle_waiting(state)
    r = -(w1 - w0)
    return r, {"r_vehicle_diff": r, "wait_veh": w1, "queue": float(state.total_queue())}


def reward_person_diff(prev, state, car_occupancy=CAR_OCCUPANCY):
    p0, p1 = person_waiting(prev, car_occupancy), person_waiting(state, car_occupancy)
    r = -(p1 - p0)
    _, bus_p = _bus_waiting(state)
    return r, {"r_person_diff": r, "wait_person": p1, "wait_bus_person": bus_p,
               "queue": float(state.total_queue())}


REWARDS = {"vehicle": reward_vehicle, "person": reward_person, "regular": reward_regular,
           "regular_g": reward_regular_g, "regular_g2": reward_regular_g2,
           "vehicle_diff": reward_vehicle_diff, "person_diff": reward_person_diff}
