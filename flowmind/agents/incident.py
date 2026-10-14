"""Incident agent: reacts when an accident blocks a lane (anomaly detection).

Owner: incident agent owner (TBD).

What to implement:
- wants_control: detect a blocked lane, e.g. a lane with a long queue but
  no movement for "incident: halt_time" seconds, or events of kind
  "incident" from the coordinator. Return a Claim if found.
- act: stop giving green to the blocked lane (it cannot move anyway) and
  serve the other directions; avoid sending cars into the blocked area.
- explain: e.g. "north_in_0 blocked for 90 s, serving east-west".

Simple example rule:
    # if a lane has queue >= 5 and occupancy > 0.8 for a long time: Claim(0.8, ...)
"""
from __future__ import annotations

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class IncidentAgent(Specialist):
    name = "incident"

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        return None  # TODO(incident owner)

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        self.last_reason = "stub: keep current phase"
        return self.keep_phase(state, allowed)  # TODO(incident owner)
