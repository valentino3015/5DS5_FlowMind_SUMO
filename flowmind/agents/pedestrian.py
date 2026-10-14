"""Pedestrian agent: gives green to pedestrians who have waited too long.

Owner: pedestrian agent owner (TBD), who also owns control/shield.py.

What to implement:
- wants_control: look at state.crossings. If a crossing has waiting people
  (at least "pedestrian: min_waiting") and max_wait above
  "pedestrian: max_wait", return a Claim.
- act: return the green phase that lets that crossing walk:
  self.phase_for_crossing(state, crossing.crossing_id, allowed) (None = not allowed now).
- explain: e.g. "4 pedestrians waited 80 s at :J1_c0".

Simple example rule:
    # if any crossing has max_wait > 60 s: Claim(0.6, ...)
"""
from __future__ import annotations

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class PedestrianAgent(Specialist):
    name = "pedestrian"

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        return None  # TODO(pedestrian owner)

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        self.last_reason = "stub: keep current phase"
        return self.keep_phase(state, allowed)  # TODO(pedestrian owner)
