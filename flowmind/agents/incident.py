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
    4. act        : PPO picks the green phase (Notion: "RL for the response");
                    without a model, the rule serves the longest queue that can
                    move (the baseline). Never a phase whose lanes are all
                    blocked, never one feeding a saturated exit.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class IncidentAgent(Specialist):
    """Detects lane blockages and keeps the junction flowing around them."""

    name = "incident"

    def __init__(self, params: dict | None = None, movements: dict | None = None,
                 model: object | None = None):
        """params: the "incident" section of config.yaml.

        movements: {phase: [(incoming lane, outgoing lane), ...]} from
        env.get_phase_movements(tls_id). Optional: when it is missing, the
        box-blocking check is simply skipped.

        model: a trained PPO policy for the response (act). If None and
        params["use_ppo"] is true, it is loaded from params["model_path"].
        Without a model, act() uses the rule-based response (the baseline).
        """
        super().__init__(params)
        self.movements = movements or {}
        self.model = model
        if self.model is None and self.params.get("use_ppo") and "model_path" in self.params:
            path = Path(self.params["model_path"])
            if path.is_file():
                try:
                    from stable_baselines3 import PPO  # lazy, like flow.py
                    self.model = PPO.load(str(path))
                except Exception:
                    self.model = None
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
        """PPO response if a model is loaded, otherwise the rule (baseline).

        PPO may only pick a safe phase; anything else falls back to the rule.
        """
        if self.model is not None:
            pred, _ = self.model.predict(self.encode(state), deterministic=True)
            action = int(pred)
            safe = self.safe_phases(state, allowed)
            if action < len(safe) and safe[action]:
                self.last_reason = f"PPO response: phase {action}"
                return action
            phase = self.rule_act(state, events, allowed)
            self.last_reason = f"PPO chose unsafe phase {action}, rule used: {self.last_reason}"
            return phase
        return self.rule_act(state, events, allowed)

    def rule_act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        """Rule-based response, the baseline PPO must beat: serve the longest safe queue."""
        safe = self.safe_phases(state, allowed)
        blocked = set(self._active_blocks(state.time))
        hold = float(self.params.get("hold_green", 20))
        if safe[state.phase] and state.time_in_phase < hold:
            # each switch costs 5 s of yellow + all-red: do not flip every tick
            self.last_reason = f"incident: keeping safe phase {state.phase} ({state.time_in_phase:.0f}s < {hold:.0f}s)"
            return state.phase
        best, best_score = None, -1

        for p in range(state.num_phases):
            if not safe[p]:
                continue
            lanes = set(state.phase_lanes[p])
            score = sum(l.queue for l in state.lanes if l.lane_id in lanes - blocked)
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
    # 5. What PPO sees (one block of numbers per phase)
    # ------------------------------------------------------------------
    def safe_phases(self, state: JunctionState, allowed: list[bool]) -> list[bool]:
        """Phases allowed by the shield that are not entirely blocked (idea 3)
        and do not feed a saturated exit (idea 5).

        A phase that also serves healthy lanes stays allowed: its green is not
        wasted. Forbidding it starved the healthy lanes and lost to fixed time.
        """
        blocked = set(self._active_blocks(state.time))
        saturated = self._saturated_out_lanes(state)
        return [
            allowed[p]
            and not set(state.phase_lanes[p]) <= blocked
            and not self._feeds_saturated(p, saturated)
            for p in range(state.num_phases)
        ]

    def encode(self, state: JunctionState) -> np.ndarray:
        """Fixed-size observation, so one PPO policy serves every junction.

        For each phase slot (max_phases of them): queue and waiting time it
        would serve on lanes that can move, serves a blocked lane, feeds a
        saturated exit, is the current phase, exists. Then time in phase.
        """
        n = int(self.params.get("max_phases", 4))
        norm_q = float(self.params.get("norm_queue", 20))
        norm_w = float(self.params.get("norm_wait", 300))
        blocked = set(self._active_blocks(state.time))
        saturated = self._saturated_out_lanes(state)
        lanes = {l.lane_id: l for l in state.lanes}
        obs = []
        for p in range(n):
            if p >= state.num_phases:
                obs += [0.0] * 6
                continue
            served = [lanes[i] for i in state.phase_lanes[p] if i in lanes and i not in blocked]
            obs += [
                min(sum(l.queue for l in served) / norm_q, 5.0),
                min(sum(l.waiting_time for l in served) / norm_w, 5.0),
                float(bool(set(state.phase_lanes[p]) & blocked)),
                float(self._feeds_saturated(p, saturated)),
                float(p == state.phase),
                1.0,
            ]
        obs.append(min(state.time_in_phase / norm_w, 5.0))
        return np.array(obs, dtype=np.float32)

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