"""Incident agent: detects a blocked lane and routes traffic around it.

Owner: Donia.
Covers IDEAS.md section 5 (BO2 gridlock prevention, BO7 anomaly detection).

Rule-based only: imports are limited to flowmind.contracts and the standard
library, as required by Guide.md rule 4.

How it works, every 5 s:
    1. _remember  : store vehicle count and "had green" for each lane
    2. _detect    : a lane that is full, gets green, and no vehicle leaves is
                    stuck; stuck for halt_time = blocked; vehicles leaving = released
    3. wants_control : claim control while a blockage is recently confirmed,
                    then let go so the lane gets a green that re-tests it
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
        self.confirmed_at: dict[str, float] = {}  # last time the lane was seen stuck on green

    # ------------------------------------------------------------------
    # 1. Memory: 24 steps = 2 minutes of history per lane
    # ------------------------------------------------------------------
    def _remember(self, state: JunctionState) -> None:
        green_lanes = set(state.phase_lanes[state.phase])
        for lane in state.lanes:
            self.history[lane.lane_id].append(
                (state.time, lane.vehicles, lane.lane_id in green_lanes)
            )

    def _green_check(self, lane_id: str, window: float) -> tuple[float, int]:
        """Seconds of green the lane got in the window, and vehicles that left during it.

        LaneState has no speed and no departure count, so departures are
        inferred from the number of vehicles on the lane dropping between two
        steps. The halted count (queue) is not used: cars creeping forward
        behind a blockage stop counting as halted for a moment.
        """
        h = self.history[lane_id]
        recent = [r for r in h if h and r[0] >= h[-1][0] - window]
        green_time, left = 0.0, 0
        for (t0, v0, green), (t1, v1, _) in zip(recent, recent[1:]):
            if green:
                green_time += t1 - t0
                left += max(0, v0 - v1)
        return green_time, left

    # ------------------------------------------------------------------
    # 2. Detection (IDEAS.md idea 1) and de-escalation (idea 6)
    # ------------------------------------------------------------------
    def _detect(self, state: JunctionState) -> None:
        halt_time = float(self.params.get("halt_time", 60))
        min_queue = int(self.params.get("min_queue", 5))
        min_occ = float(self.params.get("min_occupancy", 0.6))
        clear_occ = float(self.params.get("clear_occupancy", 0.3))
        window = float(self.params.get("check_window", 20))
        min_green = float(self.params.get("min_green_seen", 10))
        min_left = int(self.params.get("min_departures", 2))

        for lane in state.lanes:
            lid = lane.lane_id
            green_time, left = self._green_check(lid, window)
            judged = green_time >= min_green  # the lane got a fair chance to move
            moving = judged and left >= min_left
            full = lane.queue >= min_queue and lane.occupancy > min_occ

            if moving or lane.occupancy < clear_occ or lane.queue < min_queue:
                # idea 6: vehicles leave again, or the lane emptied
                # (hysteresis: a lower threshold to release than to alert)
                self._clear(lid)
            elif full and judged:
                # full, had green, nobody left: stuck
                self.suspect_since.setdefault(lid, state.time)
                self.confirmed_at[lid] = state.time
                if state.time - self.suspect_since[lid] >= halt_time:
                    self.blocked.setdefault(lid, state.time)
            # otherwise (e.g. red light): no new evidence, keep what we know

    def _clear(self, lane_id: str) -> None:
        self.suspect_since.pop(lane_id, None)
        self.blocked.pop(lane_id, None)
        self.confirmed_at.pop(lane_id, None)

    def _active_blocks(self, now: float) -> dict[str, float]:
        """Blocked lanes confirmed recently.

        While the agent drives, a blocked lane gets no green, so the agent
        cannot see it start moving again. After recheck_after seconds without
        a new confirmation it lets go: the normal controller gives the lane a
        green, which re-tests it (still stuck -> claimed again at once).
        """
        recheck = float(self.params.get("recheck_after", 60))
        return {
            lid: t for lid, t in self.blocked.items()
            if now - self.confirmed_at.get(lid, t) <= recheck
        }

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

        active = self._active_blocks(state.time)
        if not active:
            return None

        lane_id = min(active, key=active.get)  # the oldest blockage
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
        blocked = set(self._active_blocks(state.time))
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