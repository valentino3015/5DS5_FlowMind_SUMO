"""
Rewards for the last decision tick, pure functions of two JunctionStates (no SUMO).
Each returns (reward, terms) where terms is a dict for logging.

D  reward_vehicle(prev, state) = -(W(state) - W(prev)),
   W = total accumulated waiting time of the vehicles on the junction's incoming lanes
   (sum of LaneState.waiting_time). Positive when queues are discharged.
E  reward_person(prev, state)  = same with every vehicle weighted by its occupancy:
   cars CAR_OCCUPANCY, buses their persons_on_board. Bus waiting comes from
   `approaching` (the sensor covers the whole approach), car waiting = lane total - bus.
Vehicle waiting time in SUMO counts only time below 0.1 m/s that is not a planned stop,
so a bus dwelling at a stop adds nothing.
"""
CAR_OCCUPANCY = 1.3


def _bus_waiting(state):
    """(unweighted, occupancy-weighted) waiting of buses on the incoming lanes."""
    lanes = {l.lane_id for l in state.incoming}
    w = wp = 0.0
    for b in state.approaching:
        if b.vclass == "bus" and b.lane_id in lanes:
            w += b.waiting_time
            wp += b.waiting_time * (b.persons_on_board or 0)
    return w, wp


def vehicle_waiting(state):
    return sum(l.waiting_time for l in state.incoming)


def person_waiting(state, car_occupancy=CAR_OCCUPANCY):
    total = vehicle_waiting(state)
    bus, bus_p = _bus_waiting(state)
    return car_occupancy * max(total - bus, 0.0) + bus_p


def reward_vehicle(prev, state):
    w0, w1 = vehicle_waiting(prev), vehicle_waiting(state)
    r = -(w1 - w0)
    return r, {"r_vehicle": r, "wait_veh": w1, "queue": float(state.total_queue())}


def reward_person(prev, state, car_occupancy=CAR_OCCUPANCY):
    p0, p1 = person_waiting(prev, car_occupancy), person_waiting(state, car_occupancy)
    r = -(p1 - p0)
    bus, bus_p = _bus_waiting(state)
    return r, {"r_person": r, "wait_person": p1, "wait_bus_person": bus_p,
               "queue": float(state.total_queue())}


REWARDS = {"vehicle": reward_vehicle, "person": reward_person}
