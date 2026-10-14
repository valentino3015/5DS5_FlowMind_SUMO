"""Emergency agent: gives green to an ambulance / fire truck as it approaches.

Owner: emergency agent owner (TBD).

What to implement:
- wants_control: look in state.special_vehicles for kind == "emergency"
  closer than config.yaml "emergency: detect_distance". Also look at events
  of kind "emergency" (the coordinator warns you before it arrives).
  If found, return Claim(urgency close to 1, reason).
- act: return the green phase that serves the vehicle's lane:
  self.phase_for_lane(state, vehicle.lane_id, allowed) (None = not allowed now).
- explain: e.g. "green for amb_1, 80 m away on north_in_0".

Simple example rule:
    # if an emergency vehicle is within 200 m: Claim(1.0, ...)
    # act: the phase that serves its lane (if allowed), else keep_phase
"""
from __future__ import annotations

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class EmergencyAgent(Specialist):
    name = "emergency"

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        return None  # TODO(emergency owner)

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        self.last_reason = "stub: keep current phase"
        return self.keep_phase(state, allowed)  # TODO(emergency owner)
