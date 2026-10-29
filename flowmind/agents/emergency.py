"""Emergency agent: gives green to an ambulance / fire truck as it approaches.

Owner: Hadyle.

config.yaml section "emergency":
    detect_distance:   ask for control when an emergency vehicle is closer than this (m)
    preempt_eta_s:     ETA (s) under which the claim has maximum urgency
    warning_eta_s:     react to a coordinator warning when its ETA is below this (s)
    hold_after_pass_s: keep the same green this long after the vehicle disappears (s)
    min_speed_mps:     speed floor for the ETA, so a stopped vehicle has a finite ETA
    stall_patience_s:  vehicle stopped this long under a phase -> try another phase (s)
    max_stall_s:       vehicle stopped this long -> give up control until it moves (s)
"""
from __future__ import annotations

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class EmergencyAgent(Specialist):
    name = "emergency"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._lane: str | None = None
        self._veh = None                      # emergency vehicle served this step (if any)
        self._why = ""
        self._hold_lane: str | None = None
        self._hold_until = -1.0
        self._last_moved: dict[str, float] = {}   # vehicle -> last time it was moving
        self._ignored: set[str] = set()           # vehicles stuck too long: no control
        self._stall: dict[str, tuple[int, float]] = {}  # vehicle -> (phase, stopped since)
        self._skip: dict[str, set[int]] = {}      # vehicle -> phases that did not help

    def _p(self, key: str, default: float) -> float:
        return self.params.get(key, default)

    def _eta(self, distance: float, speed: float) -> float:
        return distance / max(speed, self._p("min_speed_mps", 3.0))

    def _track(self, v, now: float) -> None:
        """Remember when each emergency vehicle last moved; give up if stuck for too long."""
        last = self._last_moved.setdefault(v.vehicle_id, now)
        if v.speed >= 1.0:
            self._last_moved[v.vehicle_id] = now
            self._ignored.discard(v.vehicle_id)
        elif now - last > self._p("max_stall_s", 90.0):
            self._ignored.add(v.vehicle_id)

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        self._lane = None
        self._veh = None

        for v in state.special_vehicles:
            if v.kind == "emergency":
                self._track(v, state.time)

        # 1. An emergency vehicle is close: serve the one with the smallest ETA.
        close = [
            (self._eta(v.distance, v.speed), v)
            for v in state.special_vehicles
            if v.kind == "emergency"
            and v.distance <= self._p("detect_distance", 200.0)
            and v.vehicle_id not in self._ignored
        ]
        if close:
            eta, veh = min(close, key=lambda item: item[0])
            self._lane = veh.lane_id
            self._veh = veh
            self._hold_lane = veh.lane_id
            self._hold_until = state.time + self._p("hold_after_pass_s", 5.0)
            self._why = f"{veh.vehicle_id} is {veh.distance:.0f} m away on {veh.lane_id} (ETA {eta:.0f} s)"
            urgency = 1.0 if eta <= self._p("preempt_eta_s", 20.0) else 0.8
            return Claim(urgency=urgency, reason=self._why)

        # 2. It just passed: keep the same green a few seconds so the road clears.
        if self._hold_lane is not None and state.time < self._hold_until:
            self._lane = self._hold_lane
            self._why = "holding green while the emergency vehicle clears the junction"
            return Claim(urgency=0.6, reason=self._why)

        # 3. Coordinator warning: a vehicle is coming from another junction.
        for event in events:
            if event.kind != "emergency" or event.expires_at <= state.time:
                continue
            eta = float(event.data.get("eta", float("inf")))
            if eta <= self._p("warning_eta_s", 30.0):
                self._lane = event.data.get("lane_id")
                self._why = f"coordinator warning: emergency vehicle arrives in {eta:.0f} s"
                return Claim(urgency=0.7, reason=self._why)

        # 4. Nothing to do.
        self._hold_lane = None
        return None

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        veh = self._veh
        phase = None

        if veh is not None and self._lane:
            cands = [p for p in range(state.num_phases)
                     if self._lane in state.phase_lanes[p] and allowed[p]]
            if cands:
                skip = self._skip.setdefault(veh.vehicle_id, set())
                cur = state.phase
                if veh.speed < 0.5 and veh.distance > 2.0:
                    since = self._stall.get(veh.vehicle_id)
                    if since is None or since[0] != cur:
                        self._stall[veh.vehicle_id] = (cur, state.time)   # (re)start the timer
                    elif state.time - since[1] >= self._p("stall_patience_s", 15.0):
                        skip.add(cur)                                     # this phase does not help
                else:
                    self._stall.pop(veh.vehicle_id, None)
                    skip.clear()
                options = [p for p in cands if p not in skip]
                if not options:
                    skip.clear()
                    options = cands
                phase = cur if cur in options else options[0]
        elif self._lane:
            phase = self.phase_for_lane(state, self._lane, allowed)

        if phase is not None:
            self.last_reason = f"green for phase {phase}: {self._why}"
            return phase

        phase = self.keep_phase(state, allowed)
        self.last_reason = f"keep phase {phase}: {self._why} (no allowed phase for the lane)"
        return phase