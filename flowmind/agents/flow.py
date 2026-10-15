"""Flow agent: handles normal traffic.

Uses a trained PPO or DQN reinforcement learning model if available,
or falls back to a spillback-aware queue-draining heuristic.

Owner: Ahmed (flow agent owner). Training script: scripts/train.py.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class FlowAgent(Specialist):
    """Specialist responsible for normal, everyday traffic flow optimization."""

    name = "flow"

    def __init__(self, params: dict | None = None, model: object | None = None):
        super().__init__(params)
        self.urgency = float(self.params.get("urgency", 0.1))
        self.max_queue = float(self.params.get("max_queue", 25.0))
        self.max_wait = float(self.params.get("max_wait", 120.0))
        # Max downstream occupancy before heuristic avoids switching to a phase
        self.spillback_threshold = float(self.params.get("spillback_threshold", 0.80))
        self.model = model

        # Lazy-load a trained model from config path
        if self.model is None and "model_path" in self.params:
            model_path = Path(self.params["model_path"])
            if model_path.is_file():
                try:
                    # Support both PPO and DQN saved models via stable_baselines3
                    from stable_baselines3 import PPO, DQN
                    try:
                        self.model = PPO.load(str(model_path))
                    except Exception:
                        self.model = DQN.load(str(model_path))
                except Exception:
                    self.model = None

    # ------------------------------------------------------------------
    # Specialist contract
    # ------------------------------------------------------------------

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        """The flow agent is the default controller; always returns low urgency."""
        return Claim(urgency=self.urgency, reason="normal traffic flow optimization")

    def encode_state(self, state: JunctionState) -> np.ndarray:
        """Vectorize a JunctionState into a 1D float32 observation for the RL policy.

        Features:
        - Per incoming lane: normalized queue, normalized wait, occupancy
        - Per outgoing lane: occupancy (spillback signal)
        - Phase one-hot
        - Normalized time in current phase
        - Per phase: downstream spillback signal (max occupancy of outgoing lanes)
        """
        features: list[float] = []

        # 1. Incoming lane metrics
        for lane in state.lanes:
            features.append(min(1.0, float(lane.queue) / self.max_queue))
            features.append(min(1.0, float(lane.waiting_time) / self.max_wait))
            features.append(min(1.0, max(0.0, float(lane.occupancy))))

        # 2. Outgoing lane occupancies (downstream spillback awareness)
        for out in state.out_lanes:
            features.append(min(1.0, max(0.0, float(out.occupancy))))

        # 3. Current phase as one-hot
        for p in range(state.num_phases):
            features.append(1.0 if p == state.phase else 0.0)

        # 4. Normalized time in current phase
        features.append(min(1.0, float(state.time_in_phase) / 60.0))

        # 5. Per-phase spillback signal: max occupancy of outgoing lanes reached by each phase
        out_ids = {out.lane_id: out.occupancy for out in state.out_lanes}
        for p in range(state.num_phases):
            phase_out = state.phase_out_lanes[p] if p < len(state.phase_out_lanes) else ()
            occ_vals = [out_ids[l] for l in phase_out if l in out_ids]
            features.append(max(occ_vals) if occ_vals else 0.0)

        return np.array(features, dtype=np.float32)

    def _phase_is_spillback_blocked(self, state: JunctionState, phase: int) -> bool:
        """Return True if all outgoing lanes for this phase are above the spillback threshold."""
        if phase >= len(state.phase_out_lanes):
            return False
        out_ids = {out.lane_id: out.occupancy for out in state.out_lanes}
        phase_out = [l for l in state.phase_out_lanes[phase] if l in out_ids]
        if not phase_out:
            return False
        return all(out_ids[l] >= self.spillback_threshold for l in phase_out)

    def heuristic_act(self, state: JunctionState, allowed: list[bool]) -> int:
        """Spillback-aware greedy fallback: pick the allowed phase with the highest
        queue, but skip any phase whose downstream lanes are fully jammed.
        """
        best_phase = self.keep_phase(state, allowed)
        max_queue = -1

        for p in range(state.num_phases):
            if not allowed[p]:
                continue
            # Skip phase if downstream is blocked (would cause box-blocking)
            if self._phase_is_spillback_blocked(state, p):
                continue
            served_lanes = set(state.phase_lanes[p]) if p < len(state.phase_lanes) else set()
            q = sum(lane.queue for lane in state.lanes if lane.lane_id in served_lanes)
            if q > max_queue:
                max_queue = q
                best_phase = p

        if max_queue == -1:
            # All phases blocked by spillback; keep current phase to avoid releasing into gridlock
            self.last_reason = f"heuristic: all phases spillback-blocked, holding phase {best_phase}"
        else:
            self.last_reason = f"heuristic: phase {best_phase} has highest queue ({max_queue})"

        return best_phase

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        """Pick which green phase should be active now."""
        if self.model is not None:
            try:
                obs = self.encode_state(state)
                pred, _ = self.model.predict(obs, deterministic=True)
                action = int(pred)

                # Safety: mask spillback phases even when the RL model picks them
                if allowed[action] and not self._phase_is_spillback_blocked(state, action):
                    self.last_reason = f"RL model chose phase {action}"
                    return action

                if allowed[action] and self._phase_is_spillback_blocked(state, action):
                    self.last_reason = (
                        f"RL chose phase {action} but downstream is blocked "
                        f"(spillback), falling back to heuristic"
                    )
                    return self.heuristic_act(state, allowed)

                self.last_reason = f"RL chose phase {action} (shield blocked), kept current"
                return self.keep_phase(state, allowed)

            except Exception:
                # Fall back gracefully to heuristic if model prediction errors
                return self.heuristic_act(state, allowed)

        return self.heuristic_act(state, allowed)
