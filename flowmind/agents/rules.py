"""Emission-only priority score; traffic data is used solely for constraints.

Prioritize the permitted direction with the highest normalized current emission
burden. This reactive rule does not predict the consequences of a green phase
and cannot guarantee lower emissions. No trained model is used.
"""
from __future__ import annotations
from math import isfinite
from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class RuleBasedAgent(Specialist):
    name = "rules"

    def __init__(self, params=None):
        super().__init__(params)
        self.min_green = float(self.params.get("min_green", 10))
        self.max_green = float(self.params.get("max_green", 60))
        self.max_red = float(self.params.get("max_red", 90))
        self.weights = {key: float(self.params.get(key + "_weight", 1)) for key in ("co2", "nox", "fuel", "pm")}
        if any(not isfinite(v) or v < 0 for v in self.weights.values()) or not any(self.weights.values()):
            raise ValueError("Au moins un poids environnemental doit être positif")
        self.switch_margin = float(self.params.get("switch_margin", .05))
        self.blocked_occupancy = float(self.params.get("blocked_occupancy", .8))
        self.waiting_since = {}
        self.last_time = None

    def wants_control(self, state: JunctionState, events: list[Event]):
        return Claim(.1, "Priorité environnementale CO2, NOx, carburant et PM")

    def emission_scores(self, state):
        """Dimensionless weighted shares of junction-wide rates; no traffic terms."""
        from math import isfinite
        rates = {e.lane_id: e for e in state.lane_emissions}
        if any(lane.lane_id not in rates for lane in state.lanes):
            raise ValueError("Mesures d’émissions manquantes pour les voies entrantes")
        for rate in rates.values():
            if any(not isfinite(getattr(rate, k)) or getattr(rate, k) < 0 for k in self.weights):
                raise ValueError("Mesure d’émissions invalide")
        total = {key: sum(getattr(rates[lane.lane_id], key) for lane in state.lanes) for key in self.weights}
        weight_sum = sum(self.weights.values())
        return {p: sum(self.weights[key] * sum(getattr(rates[lane], key) for lane in set(lanes) if lane in rates)
                       / total[key] for key in self.weights if total[key] > 0) / weight_sum
                for p, lanes in enumerate(state.phase_lanes)}

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        if len(allowed) != state.num_phases or not any(allowed):
            raise ValueError("Au moins une phase doit être autorisée")
        if self.last_time is not None and state.time < self.last_time:
            self.waiting_since.clear()  # Environment reset.
        self.last_time = state.time
        current_lanes = set(state.phase_lanes[state.phase])
        for lane in state.lanes:
            if lane.queue > 0 and lane.lane_id not in current_lanes:
                self.waiting_since.setdefault(lane.lane_id, state.time)
            else:
                self.waiting_since.pop(lane.lane_id, None)
        out = {lane.lane_id: lane for lane in state.out_lanes}
        candidates, scores, red_ages = [], self.emission_scores(state), {}
        for phase in range(state.num_phases):
            if not allowed[phase]:
                continue
            exits = state.phase_out_lanes[phase] if phase < len(state.phase_out_lanes) else ()
            downstream = [out[lane] for lane in exits if lane in out]
            if downstream and all(lane.occupancy >= self.blocked_occupancy for lane in downstream):
                continue
            lanes = [lane for lane in state.lanes if lane.lane_id in state.phase_lanes[phase]]
            candidates.append(phase)
            red_ages[phase] = max((state.time-self.waiting_since[lane.lane_id]
                                   for lane in lanes if lane.lane_id in self.waiting_since), default=0)
        if state.in_transition:
            choice, reason = self.keep_phase(state, allowed), "Transition en cours"
        elif allowed[state.phase] and state.time_in_phase < self.min_green:
            choice, reason = state.phase, "Durée minimale du vert à respecter"
        elif not candidates:
            choice, reason = self.keep_phase(state, allowed), "Sorties saturées : conserver la phase autorisée"
        else:
            overdue = [p for p in candidates if p != state.phase and red_ages[p] >= self.max_red]
            alternatives = [p for p in candidates if p != state.phase]
            if overdue:
                choice = max(overdue, key=lambda p: (red_ages[p], scores[p]))
                reason = "Priorité à une direction qui attend depuis longtemps"
            elif state.time_in_phase >= self.max_green and alternatives:
                choice = max(alternatives, key=lambda p: scores[p])
                reason = "Durée maximale du vert atteinte"
            else:
                choice = max(candidates, key=lambda p: (scores[p], p == state.phase))
                if state.phase in candidates and scores[choice] <= scores[state.phase] + self.switch_margin:
                    choice = state.phase
                    reason = "Gain insuffisant pour justifier un changement de feu"
                else:
                    reason = "Priorité aux émissions normalisées CO2/NOx/carburant/PM"
        self.last_reason = f"{reason} ; phase {choice} ; score environnemental {scores.get(choice, 0):.4f}"
        return choice
