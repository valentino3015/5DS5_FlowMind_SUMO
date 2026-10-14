"""Bus agent: gives priority to a late bus.

Owner: bus agent owner (TBD).

What to implement:
- wants_control: look in state.special_vehicles for kind == "bus" whose
  delay is above config.yaml "bus: min_delay" and distance below
  "bus: detect_distance". Urgency can grow with the delay.
- act: return the green phase that serves the bus's lane (extend the
  current green if the bus is already on it).
- explain: e.g. "green for late bus_12 (70 s late, 120 m away)".

Simple example rule:
    # if a bus is more than 60 s late and less than 150 m away:
    #     Claim(min(1.0, delay / 300), "bus_12 is 70 s late")
"""
from __future__ import annotations

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class BusAgent(Specialist):
    name = "bus"

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        return None  # TODO(bus owner)

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        self.last_reason = "stub: keep current phase"
        return self.keep_phase(state, allowed)  # TODO(bus owner)
