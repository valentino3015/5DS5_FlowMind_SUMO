"""Flow agent: handles normal traffic.

Uses a trained PPO or DQN reinforcement learning model if available,
or falls back to a pressure-aware, starvation-proof heuristic.

Owner: Ahmed (flow agent owner). Training script: scripts/train.py.

Observation vector (per junction, all features in [0, 1]):
  - EMA-smoothed queue per incoming lane          (N_in features)
  - Normalized wait time per incoming lane         (N_in features)
  - Occupancy per incoming lane                    (N_in features)
  - dQ/dt queue rate-of-change per incoming lane   (N_in features, clipped to [-1, 1])
  - Outgoing lane occupancy                        (N_out features)
  - Phase one-hot                                  (N_phases features)
  - Normalized time in current phase               (1 feature)
  - Per-phase downstream spillback signal           (N_phases features)

Heuristic V2 (stateless — safe for multi-junction shared agent):
  - Pressure scoring: queue × (1 + wait/max_wait) — joint queue×delay penalty
  - Starvation multiplier: non-current phases get a bonus that grows with
    state.time_in_phase (uses the per-junction phase timer already in JunctionState,
    so it is NEVER shared/corrupted across junctions)
  - Spillback: hard skip unless ALL allowed phases are blocked
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

        # EMA smoothing: alpha=0.25 → smoother queue signals (less step-to-step noise)
        self._ema_alpha: float = float(self.params.get("ema_alpha", 0.25))
        self._ema_queues: dict[str, float] = {}

        # dQ/dt feature: store the previous step's EMA-smoothed queue per lane
        self._prev_ema_queues: dict[str, float] = {}

        # Lazy-load a trained model from config path
        if self.model is None and "model_path" in self.params:
            model_path = Path(self.params["model_path"])
            if model_path.is_file():
                try:
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

    def reset_observation_state(self) -> None:
        """Clear EMA and dQ/dt history. Call at the start of each episode."""
        self._ema_queues.clear()
        self._prev_ema_queues.clear()

    def encode_state(self, state: JunctionState) -> np.ndarray:
        """Vectorize a JunctionState into a 1D float32 observation for the RL policy.

        Features (all normalised to [0, 1] unless noted):
        1. EMA-smoothed queue per incoming lane          (N_in)
        2. Normalized wait time per incoming lane         (N_in)
        3. Occupancy per incoming lane                    (N_in)
        4. dQ/dt queue rate-of-change per incoming lane   (N_in, clipped [-1, 1])
        5. Outgoing lane occupancy (spillback signal)     (N_out)
        6. Phase one-hot                                  (N_phases)
        7. Normalized time in current phase               (1)
        8. Per-phase downstream spillback signal          (N_phases)
        """
        features: list[float] = []

        # 1 & 4. EMA-smoothed queues + dQ/dt rate of change
        ema_now: dict[str, float] = {}
        dqdt_vals: list[float] = []
        wait_vals: list[float] = []
        occ_vals_in: list[float] = []

        for lane in state.lanes:
            raw_q = min(1.0, float(lane.queue) / self.max_queue)

            # EMA smoothing: blend new observation with running average
            prev_ema = self._ema_queues.get(lane.lane_id, raw_q)
            ema_q = self._ema_alpha * raw_q + (1.0 - self._ema_alpha) * prev_ema
            ema_now[lane.lane_id] = ema_q

            # dQ/dt: change since last step (clipped to [-1, 1])
            prev_q = self._prev_ema_queues.get(lane.lane_id, ema_q)
            delta_q = float(np.clip(ema_q - prev_q, -1.0, 1.0))

            features.append(float(ema_q))
            wait_vals.append(min(1.0, float(lane.waiting_time) / self.max_wait))
            occ_vals_in.append(min(1.0, max(0.0, float(lane.occupancy))))
            dqdt_vals.append(delta_q)

        # Update EMA history for next step
        self._prev_ema_queues = dict(self._ema_queues)  # snapshot of previous step
        self._ema_queues.update(ema_now)

        # 2. Wait times
        features.extend(wait_vals)

        # 3. Incoming occupancies
        features.extend(occ_vals_in)

        # 4. dQ/dt (shifted from [-1,1] to [0,1] so the network sees a consistent positive range)
        features.extend((v + 1.0) / 2.0 for v in dqdt_vals)

        # 5. Outgoing lane occupancies (downstream spillback awareness)
        for out in state.out_lanes:
            features.append(min(1.0, max(0.0, float(out.occupancy))))

        # 6. Current phase as one-hot
        for p in range(state.num_phases):
            features.append(1.0 if p == state.phase else 0.0)

        # 7. Normalized time in current phase
        features.append(min(1.0, float(state.time_in_phase) / 60.0))

        # 8. Per-phase spillback signal: max occupancy of outgoing lanes reached by each phase
        out_occ = {out.lane_id: out.occupancy for out in state.out_lanes}
        for p in range(state.num_phases):
            phase_out = state.phase_out_lanes[p] if p < len(state.phase_out_lanes) else ()
            occ_list = [out_occ[l] for l in phase_out if l in out_occ]
            features.append(max(occ_list) if occ_list else 0.0)

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
        """Pressure-based heuristic V2 — fully stateless, per-junction starvation fix.

        Scoring for each candidate phase p:
          pressure(p)   = Σ queue_i × (1 + wait_i / max_wait)
                          [lanes with BOTH high queue AND high wait score highest]

          starvation_k  = min(3.0,  1 + time_in_phase / 30.0)
                          [grows 1× at 0s → 2× at 30s → 3× at 60s]

          score(p)      = pressure(p) × starvation_k   (if p ≠ current phase)
                        = pressure(p)                   (if p == current phase)

        Design rationale for starvation_k:
          Using state.time_in_phase (already tracked per-junction in env.py) keeps
          this completely stateless. A shared counter (like the broken V1 attempt)
          gets corrupted when the same FlowAgent handles 9 junctions sequentially
          in one step — the counter for junction 1's phase pollutes junction 4's score.
          state.time_in_phase has no such problem: it belongs to a single junction.

        Spillback phases are hard-skipped; if ALL phases are blocked, the highest-
        pressure blocked phase is chosen as a least-bad fallback.
        """
        best_phase = -1
        max_score = -1.0
        spillback_fallback: list[tuple[float, int]] = []  # (score, p) for blocked phases

        # Starvation multiplier: grows with how long the current phase has held green.
        # Completely stateless — derived entirely from per-junction JunctionState.
        starvation_k = min(3.0, 1.0 + float(state.time_in_phase) / 30.0)

        for p in range(state.num_phases):
            if not allowed[p]:
                continue

            served_lanes = set(state.phase_lanes[p]) if p < len(state.phase_lanes) else set()
            lanes = [lane for lane in state.lanes if lane.lane_id in served_lanes]

            # Pressure: each lane contributes queue × (1 + normalised_wait)
            pressure = sum(
                float(l.queue) * (1.0 + min(1.0, float(l.waiting_time) / self.max_wait))
                for l in lanes
            )

            # Boost non-current phases by the starvation multiplier to prevent monopoly
            score = pressure if p == state.phase else pressure * starvation_k

            if self._phase_is_spillback_blocked(state, p):
                spillback_fallback.append((score, p))
                continue  # hard-skip spillback phases from main ranking

            if score > max_score:
                max_score = score
                best_phase = p

        if best_phase < 0:
            # All allowed phases are spillback-blocked — pick the least bad
            if spillback_fallback:
                best_phase = max(spillback_fallback, key=lambda x: x[0])[1]
                self.last_reason = f"heuristic: all spillback-blocked, least-bad phase {best_phase}"
            else:
                best_phase = self.keep_phase(state, allowed)
                self.last_reason = f"heuristic: no allowed phase, keeping {best_phase}"
        else:
            self.last_reason = (
                f"heuristic: phase {best_phase} "
                f"(score {max_score:.2f}, starvation_k={starvation_k:.2f})"
            )

        return best_phase

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        """Pick which green phase should be active now."""
        if self.model is not None:
            try:
                obs = self.encode_state(state)
                pred, _ = self.model.predict(obs, deterministic=True)
                action = int(pred)

                # Starvation / degenerate model guard:
                # If RL keeps current phase on an empty road while cross-traffic waits,
                # fall back to heuristic so traffic never sits stranded at a red.
                curr_lanes = (
                    set(state.phase_lanes[state.phase])
                    if state.phase < len(state.phase_lanes) else set()
                )
                curr_q = sum(l.queue for l in state.lanes if l.lane_id in curr_lanes)
                other_q = sum(l.queue for l in state.lanes if l.lane_id not in curr_lanes)
                other_wait = max(
                    (l.waiting_time for l in state.lanes if l.lane_id not in curr_lanes),
                    default=0.0,
                )

                if (
                    action == state.phase
                    and curr_q == 0
                    and (other_q >= 2 or other_wait >= 20.0)
                    and state.time_in_phase >= 15.0
                ):
                    self.last_reason = (
                        f"RL kept phase {action} on empty road, "
                        f"heuristic switched to cross-traffic ({other_q} cars waiting)"
                    )
                    return self.heuristic_act(state, allowed)

                # Safety: mask spillback phases even when RL picks them
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
                return self.heuristic_act(state, allowed)

        return self.heuristic_act(state, allowed)
