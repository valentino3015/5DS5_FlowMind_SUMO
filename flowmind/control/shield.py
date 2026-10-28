"""Validate proposed signal phases against hard safety constraints."""
from __future__ import annotations

from dataclasses import dataclass

from flowmind.contracts import Crossing, JunctionState


@dataclass(frozen=True)
class ShieldResult:
    """The safe action produced by the shield and its audit explanation."""

    action: int
    overridden: bool
    reason: str


class SafetyShield:
    """Apply hard signal and pedestrian-safety constraints to phase choices."""

    def __init__(
        self,
        signal_params: dict | None = None,
        shield_params: dict | None = None,
    ):
        signal_params = signal_params or {}
        shield_params = shield_params or {}
        self.min_green = float(signal_params.get("min_green", 10))
        self.max_green = float(signal_params.get("max_green", 60))
        self.ped_max_wait = float(shield_params.get("ped_max_wait", 90))

    def allowed_phases(self, state: JunctionState) -> list[bool]:
        """Return one boolean per green phase, with at least one phase allowed."""
        if state.num_phases <= 0:
            raise ValueError("a junction must expose at least one green phase")
        if not 0 <= state.phase < state.num_phases:
            raise ValueError("current phase is outside the phase range")

        # A transition already in progress must finish before another change.
        if state.in_transition:
            return self._only(state.num_phases, state.phase)

        # Never interrupt pedestrians who are already on a crossing.
        active = [crossing for crossing in state.crossings if crossing.on_crossing > 0]
        active_phases = self._phases_serving(state, active)
        if active_phases:
            return self._mask(state.num_phases, active_phases)

        # Honour the minimum green before considering service constraints.
        if state.time_in_phase < self.min_green:
            return self._only(state.num_phases, state.phase)

        # Max-green prevents pedestrian priority (or any other specialist)
        # from starving vehicle traffic indefinitely. Prefer another phase
        # that actually releases vehicle lanes when topology provides one.
        if state.time_in_phase >= self.max_green and state.num_phases > 1:
            vehicle_phases = {
                phase
                for phase, lanes in enumerate(state.phase_lanes)
                if phase != state.phase and lanes
            }
            if vehicle_phases:
                return self._mask(state.num_phases, vehicle_phases)
            allowed = [True] * state.num_phases
            allowed[state.phase] = False
            return allowed

        # A pedestrian at the hard wait cap must be served next, provided the
        # current green has not already reached the vehicle anti-starvation cap.
        overdue = [
            crossing
            for crossing in state.crossings
            if crossing.waiting > 0 and crossing.max_wait >= self.ped_max_wait
        ]
        if overdue:
            worst = max(overdue, key=lambda c: (c.max_wait, c.waiting))
            phases = self._phases_serving(state, [worst])
            if phases:
                return self._mask(state.num_phases, phases)

        return [True] * state.num_phases

    def check(self, state: JunctionState, proposed_action: int) -> ShieldResult:
        """Accept a legal action or replace it with a safe allowed phase."""
        allowed = self.allowed_phases(state)
        if 0 <= proposed_action < state.num_phases and allowed[proposed_action]:
            return ShieldResult(proposed_action, False, "action accepted by safety shield")

        replacement = state.phase if allowed[state.phase] else allowed.index(True)
        return ShieldResult(
            replacement,
            True,
            f"phase {proposed_action} blocked; applied safe phase {replacement}",
        )

    @staticmethod
    def _only(num_phases: int, phase: int) -> list[bool]:
        return [index == phase for index in range(num_phases)]

    @staticmethod
    def _mask(num_phases: int, phases: set[int]) -> list[bool]:
        return [index in phases for index in range(num_phases)]

    @staticmethod
    def _phases_serving(
        state: JunctionState, crossings: list[Crossing]
    ) -> set[int]:
        crossing_ids = {crossing.crossing_id for crossing in crossings}
        return {
            phase
            for phase, served in enumerate(state.phase_crossings)
            if crossing_ids.intersection(served)
        }
