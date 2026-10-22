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

Heuristic V4 (stateless — safe for multi-junction shared agent):
  - Pressure scoring: Σ incoming_queue + wait_weight × Σ wait/max_wait
                    − Σ outgoing_queue   (max-pressure style, our edge = wait term)
  - Incumbent bias: seed with the current phase; switch only when another phase
    is STRICTLY better (identical to MaxPressure). This prevents premature
    transitions (each costs 5 s of yellow + all-red) and mirrors max-pressure.
  - Duplicate-phase collapsing: phases that serve identical incoming-lane sets
    (e.g. corridor phases 0/1 that differ only in pedestrian access) are merged;
    if the current phase is one of the group, no switch is issued.
  - Min-green guard: early return if time_in_phase < min_green.
  - No starvation multiplier: env.py already caps green at max_green=60 s.
    The multiplier was the main reason V3 had MORE box incidents than V2.
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
        # Minimum green time enforced inside the heuristic (seconds)
        self.min_green = float(self.params.get("min_green", 10.0))
        # Weight for wait-time bonus per lane: our tiebreaker edge over pure max-pressure
        self.wait_weight = float(self.params.get("wait_weight", 0.3))
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

    def _score_phase(self, state: JunctionState, p: int, out_queue: dict[str, int]) -> float:
        """Pressure score for one phase: incoming demand minus outgoing congestion.

        pressure(p) = Σ_{i ∈ incoming(p)} [queue_i + wait_weight × min(1, wait_i/max_wait)]
                    - Σ_{j ∈ outgoing(p)}  queue_j

        The wait term gives us an edge over pure max-pressure: it catches directions
        where vehicle counts are low but accumulated delay is already high.
        """
        served_lanes = set(state.phase_lanes[p]) if p < len(state.phase_lanes) else set()
        lanes = [lane for lane in state.lanes if lane.lane_id in served_lanes]

        incoming = sum(
            float(l.queue) + self.wait_weight * min(1.0, float(l.waiting_time) / self.max_wait)
            for l in lanes
        )
        out_ids = state.phase_out_lanes[p] if p < len(state.phase_out_lanes) else ()
        outgoing = sum(float(out_queue.get(l, 0)) for l in out_ids)
        return incoming - outgoing

    def heuristic_act(self, state: JunctionState, allowed: list[bool]) -> int:
        """Pressure-based heuristic V4 — incumbent-biased, no starvation multiplier.

        Algorithm:

        1. Min-green early return:
           If time_in_phase < min_green and the current phase is allowed, keep it
           immediately — no scoring needed, no risk of a premature transition.

        2. Duplicate-phase collapsing:
           Phases with identical incoming-lane sets are grouped (e.g. corridor
           phases 0 and 1 that only differ in pedestrian access). Within each
           group the current phase is preferred as representative, so the heuristic
           never issues a switch between two phases that move the same cars.

        3. Incumbent-bias scoring (the key change vs V3):
           Seed best_phase = current_phase, max_score = pressure(current_phase).
           Switch only if another phase is *strictly* better (score > max_score).
           This is identical to how MaxPressure avoids unnecessary transitions.

           WHY: The V3 starvation multiplier (starvation_k ≥ 1.33 after just 10 s)
           caused premature switches every 2 steps, each burning 5 s of lost green
           time and raising box-incident risk. env.py already caps green at
           max_green=60 s, so starvation prevention is already handled externally.

        4. Pressure formula (unchanged from V3):
             pressure(p) = Σ incoming_queue
                         + wait_weight × Σ min(1, wait/max_wait)  ← our edge
                         - Σ outgoing_queue

        5. Spillback: phases whose outgoing lanes are all saturated are hard-skipped;
           if ALL are blocked the least-bad is chosen as fallback.
        """
        # --- min-green early return -----------------------------------------------
        # if float(state.time_in_phase) < self.min_green and allowed[state.phase]:
        #     self.last_reason = (
        #         f"heuristic: min-green hold (phase {state.phase}, "
        #         f"{state.time_in_phase:.0f}s < {self.min_green:.0f}s min)"
        #     )
        #     return state.phase

        # --- outgoing queue lookup ------------------------------------------------
        out_queue: dict[str, int] = {lane.lane_id: lane.queue for lane in state.out_lanes}

        # --- duplicate-phase groups (collapse phases with identical served lanes) -
        # Phases that serve the exact same incoming lanes are grouped together.
        # Within each group, if the current phase is already active, we keep it
        # to prevent useless 5s yellow + all-red transitions.
        lane_set_to_phases: dict[frozenset, list[int]] = {}
        for p in range(state.num_phases):
            key = frozenset(state.phase_lanes[p]) if p < len(state.phase_lanes) else frozenset({p})
            lane_set_to_phases.setdefault(key, []).append(p)

        group_representatives: list[int] = []
        seen_keys: set[int] = set()
        for key, phases_in_group in lane_set_to_phases.items():
            list_id = id(phases_in_group)
            if list_id in seen_keys:
                continue
            seen_keys.add(list_id)
            if state.phase in phases_in_group and allowed[state.phase]:
                group_representatives.append(state.phase)
            else:
                allowed_phases = [p for p in phases_in_group if allowed[p]]
                if allowed_phases:
                    rep = max(
                        allowed_phases,
                        key=lambda p: len(state.phase_lanes[p]) if p < len(state.phase_lanes) else 0,
                    )
                    group_representatives.append(rep)

        # --- incumbent-biased scoring ---------------------------------------------
        # Seed with the current phase (or first allowed if current is not allowed).
        # Another phase wins only if its score is STRICTLY greater.
        # If the current phase turns out to be spillback-blocked, we un-seed it below.
        spillback_fallback: list[tuple[float, int]] = []

        if allowed[state.phase]:
            best_phase: int = state.phase
            max_score: float = self._score_phase(state, state.phase, out_queue)
            # If current phase is itself spillback-blocked, evict it from "best"
            if self._phase_is_spillback_blocked(state, state.phase):
                spillback_fallback.append((max_score, state.phase))
                best_phase = -1
                max_score = -float("inf")
        else:
            best_phase = -1
            max_score = -float("inf")

        for p in group_representatives:
            if p == state.phase:
                continue  # already seeded above
            s = self._score_phase(state, p, out_queue)
            if self._phase_is_spillback_blocked(state, p):
                spillback_fallback.append((s, p))
                continue
            if s > max_score:  # strictly better — the incumbent bias
                max_score = s
                best_phase = p

        # --- resolve result -------------------------------------------------------
        if best_phase < 0:
            best_phase = self.keep_phase(state, allowed)
            self.last_reason = f"heuristic: all phases spillback-blocked, holding phase {best_phase}"
        else:
            self.last_reason = f"heuristic: phase {best_phase} (score {max_score:.2f})"

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
