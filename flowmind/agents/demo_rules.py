"""Temporary demo rule by Adem. At integration, replaced by the emergency owner's agents/emergency.py.

Owner: Adem.

DemoEmergency gives green to an ambulance, using two sources:
- an emergency vehicle on an incoming lane, closer than emergency.detect_distance
- a coordinator warning (Event "emergency") with ETA below emergency.preclear_eta:
  the junction turns green for the ambulance's lane before it arrives.
Choose it in config.yaml with  agents: {emergency: demo}  (or "team" for emergency.py).
"""
from __future__ import annotations

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class DemoEmergency(Specialist):
    name = "emergency"

    def target(self, state: JunctionState, events: list[Event]):
        """The ambulance to serve: (vehicle_id, lane_id, description), or None."""
        # 1. An ambulance already on one of our incoming lanes: the nearest one.
        near = [v for v in state.special_vehicles
                if v.kind == "emergency" and v.distance <= self.params["detect_distance"]]
        if near:
            v = min(near, key=lambda v: v.distance)
            return v.vehicle_id, v.lane_id, f"{v.distance:.0f} m away"
        # 2. A warning from the coordinator: an ambulance arrives soon.
        soon = [e for e in events if e.kind == "emergency" and e.expires_at >= state.time
                and e.data["eta"] <= self.params["preclear_eta"]]
        if soon:
            e = min(soon, key=lambda e: e.data["eta"])
            return e.data["vehicle_id"], e.data["lane_id"], f"arriving in {e.data['eta']:.0f} s (coordinator)"
        return None

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        t = self.target(state, events)
        if t is None:
            return None
        return Claim(urgency=1.0, reason=f"ambulance {t[0]} {t[2]}")

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        t = self.target(state, events)
        if t is not None:
            phase = self.phase_for_lane(state, t[1], allowed)
            if phase is not None:
                self.last_reason = f"green for ambulance {t[0]} on {t[1]}"
                return phase
            self.last_reason = f"ambulance {t[0]}: its phase is not allowed yet, keep phase"
        else:
            self.last_reason = "no ambulance: keep phase"
        return self.keep_phase(state, allowed)
