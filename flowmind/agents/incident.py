"""Incident agent: detects a blocked lane and routes traffic around it.

Owner: Donia.
Covers IDEAS.md section 5 (BO2 gridlock prevention, BO7 anomaly detection).

Rule-based only: imports are limited to flowmind.contracts and the standard
library, as required by Guide.md rule 4.

How it works, every 5 s:
    1. _remember  : store queue, occupancy and "had green" for each lane
    2. _detect    : a lane that is full, has green, and never drains is blocked
    3. wants_control : claim control once a lane has been blocked long enough
    4. act        : serve the phase that frees the most vehicles, while never
                    giving green to a blocked lane or feeding a saturated exit
"""
from __future__ import annotations

from collections import defaultdict, deque

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class IncidentAgent(Specialist):
    """Detects lane blockages and keeps the junction flowing around them."""

    name = "incident"

    def __init__(self, params: dict | None = None, movements: dict | None = None):
        """params: the "incident" section of config.yaml.

        movements: {phase: [(incoming lane, outgoing lane), ...]} from
        env.get_phase_movements(tls_id). Optional: when it is missing, the
        box-blocking check is simply skipped.
        """
        super().__init__(params)
        self.movements = movements or {}
        self.history: dict[str, deque] = defaultdict(lambda: deque(maxlen=24))
        self.suspect_since: dict[str, float] = {}
        self.blocked: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Memory: 24 steps = 2 minutes of history per lane
    # ------------------------------------------------------------------
    def _remember(self, state: JunctionState) -> None:
        green_lanes = set(state.phase_lanes[state.phase])
        for lane in state.lanes:
            self.history[lane.lane_id].append(
                (state.time, lane.queue, lane.occupancy, lane.lane_id in green_lanes)
            )

    def _is_stuck(self, lane_id: str, window: float) -> bool:
        """True if the lane had green during the window and its queue never dropped.

        LaneState has no speed, so "nothing is moving" is inferred from the
        queue failing to drain while the lane was being served.
        """
        h = self.history[lane_id]
        if not h:
            return False
        recent = [r for r in h if r[0] >= h[-1][0] - window]
        if len(recent) < 3:
            return False
        had_green = any(r[3] for r in recent)
        queues = [r[1] for r in recent]
        drained = min(queues) < queues[0] - 1  # tolerate one vehicle of noise
        return had_green and not drained

    # ------------------------------------------------------------------
    # 2. Detection (IDEAS.md idea 1) and de-escalation (idea 6)
    # ------------------------------------------------------------------
    def _detect(self, state: JunctionState) -> None:
        halt_time = float(self.params.get("halt_time", 60))
        min_queue = int(self.params.get("min_queue", 5))
        min_occ = float(self.params.get("min_occupancy", 0.8))
        clear_occ = float(self.params.get("clear_occupancy", 0.5))
        window = float(self.params.get("check_window", 20))

        for lane in state.lanes:
            lid = lane.lane_id
            suspicious = (
                lane.queue >= min_queue
                and lane.occupancy > min_occ
                and self._is_stuck(lid, window)
            )
            if suspicious:
                self.suspect_since.setdefault(lid, state.time)
                if state.time - self.suspect_since[lid] >= halt_time:
                    self.blocked.setdefault(lid, state.time)
            elif lane.occupancy < clear_occ or lane.queue < min_queue:
                # hysteresis: a lower threshold to release than to alert,
                # so the agent does not flicker around the limit
                self.suspect_since.pop(lid, None)
                self.blocked.pop(lid, None)

    # ------------------------------------------------------------------
    # 3. Claim (IDEAS.md idea 2)
    # ------------------------------------------------------------------
    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        self._remember(state)
        self._detect(state)
        urgency = float(self.params.get("urgency", 0.85))

        for e in events:
            if e.kind == "incident" and e.expires_at > state.time:
                return Claim(urgency, f"incident reported by the coordinator: {e.data}")

        if not self.blocked:
            return None

        lane_id = min(self.blocked, key=self.blocked.get)  # the oldest blockage
        held = state.time - self.suspect_since.get(lane_id, self.blocked[lane_id])
        queue = next((l.queue for l in state.lanes if l.lane_id == lane_id), 0)
        return Claim(
            urgency,
            f"Incident suspected on {lane_id}: halted for {held:.0f}s with queue {queue}",
        )

    # ------------------------------------------------------------------
    # 4. Act (IDEAS.md ideas 3 and 5)
    # ------------------------------------------------------------------
    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        blocked = set(self.blocked)
        saturated = self._saturated_out_lanes(state)
        best, best_score = None, -1

        for p in range(state.num_phases):
            if not allowed[p]:
                continue
            lanes = set(state.phase_lanes[p])
            if lanes & blocked:
                continue  # idea 3: never serve a lane that cannot move
            if self._feeds_saturated(p, saturated):
                continue  # idea 5: never discharge into a full exit
            score = sum(l.queue for l in state.lanes if l.lane_id in lanes)
            if score > best_score:
                best, best_score = p, score

        if best is None:
            self.last_reason = "incident: no safe alternative phase, keeping current green"
            return self.keep_phase(state, allowed)

        self.last_reason = (
            f"{', '.join(sorted(blocked)) or 'incident'} blocked -> phase {best} "
            f"({best_score} vehicles waiting)"
        )
        return best

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _saturated_out_lanes(self, state: JunctionState) -> set[str]:
        limit = float(self.params.get("out_occupancy", 0.95))
        return {l.lane_id for l in state.out_lanes if l.occupancy >= limit}

    def _feeds_saturated(self, phase: int, saturated: set[str]) -> bool:
        if not self.movements or not saturated:
            return False
        return any(dst in saturated for _, dst in self.movements.get(phase, []))