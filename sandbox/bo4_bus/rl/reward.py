"""
Rewards for the last decision tick, pure functions of JunctionStates (no SUMO).
Each returns (reward, terms); terms are unscaled values for logging.

Main rewards (delay incurred this tick, as in the BO4 delay metrics):
D  reward_vehicle(prev, state) = -(halting vehicles on the incoming lanes) * dt
E  reward_person(prev, state)  = -(halting persons on the incoming lanes) * dt,
   cars weighted CAR_OCCUPANCY, buses their persons_on_board.
"Halting" = below 0.1 m/s (LaneState.halting_count). A bus dwelling at a stop is
halted but is not signal delay, so it is excluded (VehicleInfo.at_stop). Car
halting = lane halting - all halting buses; the sensor covers the whole approach,
so every bus on the incoming lanes is in `approaching`. `prev` is unused (kept for
the LearningSpecialist.reward(prev, state, events) signature).

Reference only (phase 1, not used for training):
reward_vehicle_diff / reward_person_diff = -(change in total accumulated waiting time).
Their episode sum telescopes to (waiting at start - waiting at end), ~0 for any policy.
"""
CAR_OCCUPANCY = 1.3
TICK = 5.0            # s, decision interval
HALT_SPEED = 0.1      # m/s, SUMO's halting threshold


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


REWARDS = {"vehicle": reward_vehicle, "person": reward_person,
           "vehicle_diff": reward_vehicle_diff, "person_diff": reward_person_diff}
