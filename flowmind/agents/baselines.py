"""Finished example agents: FixedTime, MaxPressure and GreenWave.

Owner: Adem. These are the examples to copy when you write your own agent.
They are also the baselines we compare our system against.
"""
from __future__ import annotations

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class FixedTime(Specialist):
    """Classic timer: each green phase gets the same time, in order 0, 1, 2, ...

    config.yaml section "fixed_time":
        green: seconds of green per phase
    """

    name = "fixed_time"

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        # A baseline is always ready to drive the junction, with no urgency.
        return Claim(urgency=0.0, reason="fixed timer")

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        green = self.params["green"]

        # 1. Phase not finished yet: keep it.
        if state.time_in_phase < green and allowed[state.phase]:
            self.last_reason = f"phase {state.phase} has {green - state.time_in_phase:.0f} s left"
            return state.phase

        # 2. Time is up: go to the next allowed phase (wrap around after the last one).
        for step in range(1, state.num_phases + 1):
            nxt = (state.phase + step) % state.num_phases
            if allowed[nxt]:
                self.last_reason = f"timer: phase {state.phase} -> {nxt}"
                return nxt

        # 3. Nothing else allowed (should not happen): keep the current phase.
        self.last_reason = "no other phase allowed"
        return state.phase


class MaxPressure(Specialist):
    """Give green to the phase with the highest "pressure".

    Pressure of a phase = cars waiting on the lanes it lets go
                        - cars waiting on the lanes they go to.
    A high pressure means: many cars want to go, and there is room for them.
    This is a well-known, simple and strong rule.

    movements: {phase: [(incoming lane, outgoing lane), ...]}, which moves
    get green in each phase (see contracts.example_phase_movements()).
    """

    name = "max_pressure"

    def __init__(self, params: dict | None = None, movements: dict | None = None):
        super().__init__(params)
        self.movements = movements or {}

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        return Claim(urgency=0.0, reason="max-pressure baseline")

    def pressure(self, state: JunctionState, phase: int) -> int:
        queue = {lane.lane_id: lane.queue for lane in state.lanes + state.out_lanes}
        return sum(queue.get(src, 0) - queue.get(dst, 0) for src, dst in self.movements.get(phase, []))

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        # Start with the current phase, so we only switch if another phase is strictly better.
        best = self.keep_phase(state, allowed)
        best_pressure = self.pressure(state, best)

        for phase in range(state.num_phases):
            if allowed[phase] and self.pressure(state, phase) > best_pressure:
                best, best_pressure = phase, self.pressure(state, phase)

        self.last_reason = f"phase {best} has the highest pressure ({best_pressure})"
        return best


class GreenWave(Specialist):
    """Fixed timer with an offset per junction, so a group of cars driving down the
    main road meets green lights one after the other (a "green wave").

    Static time-distance plan (backup idea 1 for core idea 6):
        offset of a junction = its position on the corridor x travel time between junctions
        travel time = link_length / speed   (200 m / 13.89 m/s = about 14.4 s)
    The phase is computed from the simulation time, so the plan stays on schedule
    even after yellow and all-red.

    config.yaml section "green_wave": green (s per phase), link_length (m), speed (m/s).
    offset: seconds this junction is shifted (run.py computes it from corridor.order).
    """

    name = "green_wave"

    def __init__(self, params: dict | None = None, offset: float = 0.0):
        super().__init__(params)
        self.offset = offset

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        return Claim(urgency=0.0, reason="green-wave timer")

    def scheduled_phase(self, state: JunctionState) -> int:
        green = self.params["green"]
        cycle = green * state.num_phases
        return int(((state.time - self.offset) % cycle) // green)

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        phase = self.scheduled_phase(state)
        if allowed[phase]:
            self.last_reason = f"green wave: phase {phase} (offset {self.offset:.1f} s)"
            return phase
        self.last_reason = f"green wave wants phase {phase}, not allowed yet: keep phase"
        return self.keep_phase(state, allowed)


def green_wave_offset(params: dict, order: list[str], tls_id: str) -> float:
    """Offset of one junction: its position on the corridor x (link_length / speed)."""
    position = order.index(tls_id) if tls_id in order else 0
    return position * params["link_length"] / params["speed"]
