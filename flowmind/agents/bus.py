"""Bus agent: conditional transit signal priority (BO4, controller C).

Owner: Firas (BO4, public transport reliability and bus bunching).

Ported from the BO4 prototype (sandbox/bo4_bus/rule_tsp.py, controller C).
On a 3x3 grid, over 20 unseen evaluation seeds, the prototype cut bunching at
the last stop by 23% and bus delay per junction by 21% versus fixed-time,
without increasing car or cross-street delay. Unconditional priority (helping
every bus) made headways LESS regular, which is why this agent only helps late
buses.

Rules, all numbers from config.yaml "bus":
1. Detect buses closer than detect_distance (m) to the stop line.
2. Lateness is headway-based: time since the previous bus of the same line
   passed THIS junction, minus headway (s). The agent remembers the last
   passage per line itself (one BusAgent per junction). A bus with no previous
   passage on record is not prioritized.
3. Claim control only for a bus later than late_threshold (s).
4. Refuse when an approach that would lose green has a queue above max_queue
   vehicles (fairness for the cross street).
5. Priority budget: at most max_hold seconds of control per bus, so one bus
   cannot hold a junction.
6. Conflicts: the latest bus wins; tie: the nearest bus.
act() gives green to the phase serving the bus lane (keeps the current green
if it already does = green extension; switches = early recall). The shield
still enforces min/max green, yellow and all-red.

Known contract gap (to discuss with Adem): SpecialVehicle.delay is filled by
env.py with SUMO's waiting time, not lateness, and the line id is not in the
contract. This agent therefore computes lateness itself and derives the line
from the vehicle id ("bus_we.3" -> "bus_we", "L1_EB_09" -> "L1_EB").
"""
from __future__ import annotations

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState, SpecialVehicle

DEFAULTS = {
    "detect_distance": 150.0,  # m
    "headway": 300.0,          # s, target headway of the bus lines in the scenario
    "late_threshold": 0.0,     # s, help buses later than this
    "max_queue": 15,           # veh, refuse if a losing approach queue exceeds this
    "max_hold": 20.0,          # s of control per bus at most
    "urgency_scale": 300.0,    # s of lateness that gives urgency 1
    "passed_margin": 20.0,     # m, slack on top of speed x step for "it crossed" (see _update_memory)
    "max_accel": 1.2,          # m/s2, bus acceleration (SUMO default): a bus leaving a stop at 0 m/s still moves
    "stopped_speed": 0.1,      # m/s, slower than this the bus stands still
    "stopped_far": 120.0,      # m, a bus standing still farther than this is dwelling at a stop: no priority yet
}


def line_of(vehicle_id: str) -> str:
    """Line id from a SUMO vehicle id: 'bus_we.3' -> 'bus_we', 'L1_EB_09' -> 'L1_EB'."""
    if "." in vehicle_id:
        return vehicle_id.split(".", 1)[0]
    head, _, tail = vehicle_id.rpartition("_")
    return head if head and tail.isdigit() else vehicle_id


class BusAgent(Specialist):
    name = "bus"

    def __init__(self, params: dict | None = None):
        super().__init__({**DEFAULTS, **(params or {})})
        self.last_passage: dict[str, float] = {}   # line -> time a bus of it crossed
        self.seen: dict[str, tuple[float, float]] = {}  # bus id -> (last distance, last speed)
        self.held: dict[str, float] = {}           # bus id -> seconds of control used
        self.last_time: float | None = None
        self.served: str | None = None             # bus currently being served
        self.target: SpecialVehicle | None = None

    # --- memory -----------------------------------------------------------
    def _update_memory(self, state: JunctionState) -> None:
        dt = 0.0 if self.last_time is None else max(0.0, state.time - self.last_time)
        if self.served is not None:
            self.held[self.served] = self.held.get(self.served, 0.0) + dt
        self.last_time = state.time
        here = {v.vehicle_id: (v.distance, v.speed) for v in state.special_vehicles if v.kind == "bus"}
        # A bus that left the approach lanes has crossed only if it could reach the stop line since
        # the last decision: last distance <= speed x step + 1/2 x max_accel x step^2 + passed_margin
        # (the acceleration term: a bus leaving a stop at 0 m/s still covers ~15 m in 5 s). A fixed
        # 30 m check missed most crossings (at 13.9 m/s a bus moves ~70 m in 5 s); no check at all
        # counted a bus that vanished far away (teleported, end of route) as crossed.
        reach_extra = 0.5 * self.params["max_accel"] * dt * dt + self.params["passed_margin"]
        for bus_id, (distance, speed) in self.seen.items():
            if bus_id not in here:
                if distance <= max(speed, 0.0) * dt + reach_extra:
                    self.last_passage[line_of(bus_id)] = state.time   # it crossed the junction
                self.held.pop(bus_id, None)
        self.seen = here
        self.served = None

    def lateness(self, bus: SpecialVehicle, now: float) -> float | None:
        last = self.last_passage.get(line_of(bus.vehicle_id))
        return None if last is None else now - last - self.params["headway"]

    def _losing_queue(self, state: JunctionState, bus: SpecialVehicle) -> int:
        """Largest queue on incoming lanes NOT served by any phase serving the bus lane."""
        serving = set()
        for lanes in state.phase_lanes:
            if bus.lane_id in lanes:
                serving.update(lanes)
        return max((l.queue for l in state.lanes if l.lane_id not in serving), default=0)

    # --- the 3 functions --------------------------------------------------
    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        self._update_memory(state)
        self.target = None
        p = self.params
        best, best_late, notes = None, None, []
        for bus in state.special_vehicles:
            if bus.kind != "bus" or bus.distance > p["detect_distance"]:
                continue
            if bus.speed < p["stopped_speed"] and bus.distance > p["stopped_far"]:
                # Standing still far from the stop line: passengers are boarding at a stop zone
                # (corridor5: 140 m before the next junction). A green now would not reach it.
                notes.append(f"{bus.vehicle_id}: stopped {bus.distance:.0f} m away (bus stop), not helped yet")
                continue
            late = self.lateness(bus, state.time)
            if late is None:
                notes.append(f"{bus.vehicle_id}: no previous bus on record")
                continue
            if late <= p["late_threshold"]:
                notes.append(f"{bus.vehicle_id}: {-late:.0f} s early, not helped")
                continue
            if self.held.get(bus.vehicle_id, 0.0) >= p["max_hold"]:
                notes.append(f"{bus.vehicle_id}: priority budget of {p['max_hold']:.0f} s used")
                continue
            q = self._losing_queue(state, bus)
            if q > p["max_queue"]:
                notes.append(f"{bus.vehicle_id}: cross queue {q} veh > {p['max_queue']}")
                continue
            if best is None or (late, -bus.distance) > (best_late, -best.distance):
                best, best_late = bus, late
        if best is None:
            self.last_reason = "; ".join(notes) if notes else "no late bus nearby"
            return None
        self.target = best
        self.served = best.vehicle_id
        urgency = min(1.0, max(0.0, best_late / p["urgency_scale"]))
        return Claim(urgency, f"{best.vehicle_id} is {best_late:.0f} s late, {best.distance:.0f} m away")

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        bus = self.target
        if bus is None:
            self.last_reason = "no late bus: keep current phase"
            return self.keep_phase(state, allowed)
        late = self.lateness(bus, state.time) or 0.0
        phase = self.phase_for_lane(state, bus.lane_id, allowed)
        if phase is None:
            self.last_reason = f"{bus.vehicle_id} late but its green is not allowed yet: keep phase"
            return self.keep_phase(state, allowed)
        action = "extend green" if phase == state.phase else "early recall"
        self.last_reason = f"{action} for {bus.vehicle_id} ({late:.0f} s late, {bus.distance:.0f} m away)"
        return phase
