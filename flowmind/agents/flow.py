"""Flow agent: handles normal traffic.

Uses a trained PPO reinforcement learning model if available,
or falls back to a greedy queue-draining heuristic.

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
        self.model = model

        # Lazy load trained model from path if provided and not already injected
        if self.model is None and "model_path" in self.params:
            model_path = Path(self.params["model_path"])
            if model_path.is_file():
                try:
                    from stable_baselines3 import PPO
                    self.model = PPO.load(str(model_path))
                except Exception:
                    self.model = None

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        """The flow agent is the default controller; it always returns low urgency."""
        return Claim(urgency=self.urgency, reason="normal traffic flow optimization")

    def encode_state(self, state: JunctionState) -> np.ndarray:
        """Vectorize a JunctionState into a 1D float32 observation for the RL policy.

        Features:
        - For each incoming lane: normalized queue (0..1), normalized wait (0..1), occupancy (0..1)
        - For each outgoing lane: occupancy (0..1)
        - Current phase one-hot: length state.num_phases
        - Normalized time in current phase (capped at 60s)
        """
        features: list[float] = []

        # 1. Incoming lane metrics
        for lane in state.lanes:
            features.append(min(1.0, float(lane.queue) / self.max_queue))
            features.append(min(1.0, float(lane.waiting_time) / self.max_wait))
            features.append(min(1.0, max(0.0, float(lane.occupancy))))

        # 2. Outgoing lane occupancies (helps detect spillback)
        for out in state.out_lanes:
            features.append(min(1.0, max(0.0, float(out.occupancy))))

        # 3. Current phase as one-hot
        for p in range(state.num_phases):
            features.append(1.0 if p == state.phase else 0.0)

        # 4. Normalized time spent in the current phase
        features.append(min(1.0, float(state.time_in_phase) / 60.0))

        return np.array(features, dtype=np.float32)

    def heuristic_act(self, state: JunctionState, allowed: list[bool]) -> int:
        """Greedy fallback: select the allowed phase serving the highest total queue."""
        best_phase = self.keep_phase(state, allowed)
        max_queue = -1

        for p in range(state.num_phases):
            if allowed[p]:
                served_lanes = set(state.phase_lanes[p])
                q = sum(lane.queue for lane in state.lanes if lane.lane_id in served_lanes)
                if q > max_queue:
                    max_queue = q
                    best_phase = p

        self.last_reason = f"heuristic: phase {best_phase} has highest queue ({max_queue})"
        return best_phase

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        """Pick which green phase should be active now."""
        if self.model is not None:
            try:
                obs = self.encode_state(state)
                pred, _ = self.model.predict(obs, deterministic=True)
                action = int(pred)
                if 0 <= action < state.num_phases and allowed[action]:
                    self.last_reason = f"PPO model chose phase {action}"
                    return action
                else:
                    self.last_reason = f"PPO chose phase {action} (shield blocked), kept current"
                    return self.keep_phase(state, allowed)
            except Exception:
                # Fall back gracefully to heuristic if model prediction errors
                return self.heuristic_act(state, allowed)

        return self.heuristic_act(state, allowed)
