"""Orchestrator (the manager): picks which agent controls a junction, every step.

Owner: Adem.

One Orchestrator per junction. Each step, decide():
1. asks every agent wants_control(state, events)
2. picks the claiming agent highest in orchestrator.priority (urgency breaks ties)
3. min_hold: an agent that just got control keeps it for orchestrator.min_hold
   seconds, unless a HIGHER-priority agent claims (an ambulance never waits)
4. if nobody claims, the fallback agent (max-pressure) drives the junction
5. the chosen agent's act() proposes a phase; the shield checks it

Extras:
- operator override (set_override): a person or a test forces a phase for a
  while; the shield still checks it
- latency: every Decision records how long decide() took (real-time check)
- what-if: an optional "shadow" agent (the fixed timer) says what it would have
  chosen from the same state, so the log shows where we differ from it
"""
from __future__ import annotations

import time as clock
from dataclasses import dataclass

from flowmind.contracts import Event, JunctionState
from flowmind.control import shield


@dataclass
class Decision:
    """What happened at one junction in one step (one row in the decision log)."""

    tls_id: str
    time: float
    agent: str           # name of the agent in control
    urgency: float       # its claim urgency (0 for the fallback)
    reason: str          # why it has control (its claim reason)
    proposed: int        # phase the agent asked for
    final: int           # phase actually applied (after the shield)
    shield_note: str     # "" if the shield changed nothing
    explanation: str     # the agent's explain(): why this phase
    latency_ms: float = 0.0                # how long decide() took (ms)
    fixed_time_would: int | None = None    # what-if: phase the fixed timer would have chosen
    claims: str = ""     # every agent that asked, "name:urgency:reason; ..." (for "why not" explanations)


def claims_text(claims: dict, rank) -> str:
    """{"flow": Claim(0.1, "normal traffic")} -> "flow:0.10:normal traffic", most important first."""
    names = sorted(claims, key=lambda n: (rank(n), -claims[n].urgency))
    return "; ".join(f"{n}:{claims[n].urgency:.2f}:{claims[n].reason}" for n in names)


class Orchestrator:
    def __init__(self, agents: list, priority: list[str], min_hold: float, fallback, shadow=None):
        """agents: Specialist objects; priority: agent names, most important first.
        shadow: optional agent (e.g. FixedTime) only used for the what-if column."""
        self.agents = {agent.name: agent for agent in agents}
        self.priority = priority
        self.min_hold = min_hold
        self.fallback = fallback
        self.shadow = shadow
        self.current = fallback.name  # agent in control now
        self.since = 0.0              # time it got control
        self.override = None          # (phase, start, end, reason) while an operator forces a phase

    def set_override(self, phase: int, start: float, duration: float, reason: str = "operator command"):
        """Force `phase` from time `start` for `duration` seconds. The shield still applies."""
        self.override = (phase, start, start + duration, reason)

    def clear_override(self):
        self.override = None

    def _override_active(self, now: float) -> bool:
        return self.override is not None and self.override[1] <= now < self.override[2]

    def rank(self, name: str) -> int:
        """0 = most important. The fallback (not in the list) is the least important."""
        return self.priority.index(name) if name in self.priority else len(self.priority)

    def decide(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> Decision:
        started = clock.perf_counter()

        # 0. Operator override beats every agent, but not the shield.
        if self._override_active(state.time):
            phase, _, end, why = self.override
            if self.current != "operator":
                self.current, self.since = "operator", state.time
            final, note = shield.check(state, phase, allowed)
            return self._finish(started, state, allowed, Decision(
                tls_id=state.tls_id, time=state.time, agent="operator", urgency=1.0,
                reason=f"{why} (until t={end:.0f} s)", proposed=phase, final=final,
                shield_note=note, explanation=f"operator override: phase {phase}"))

        # 1. Who wants control?
        claims = {}
        for name, agent in self.agents.items():
            claim = agent.wants_control(state, events)
            if claim is not None:
                claims[name] = claim

        # 2. Best claim by priority, then urgency. Nobody -> fallback.
        if claims:
            winner = min(claims, key=lambda n: (self.rank(n), -claims[n].urgency))
        else:
            winner = self.fallback.name

        # 3. min_hold: do not hand control to a LOWER-priority agent too soon.
        held = state.time - self.since
        if winner != self.current and held < self.min_hold and self.rank(winner) > self.rank(self.current):
            winner = self.current
            reason = f"keeps control for min_hold ({held:.0f}/{self.min_hold:.0f} s)"
        elif winner in claims:
            reason = claims[winner].reason
        else:
            reason = "nobody asked for control: default controller"

        if winner != self.current:
            self.current, self.since = winner, state.time

        # 4. The chosen agent acts, the shield checks.
        agent = self.agents.get(winner, self.fallback)
        proposed = agent.act(state, events, allowed)
        final, note = shield.check(state, proposed, allowed)
        return self._finish(started, state, allowed, Decision(
            tls_id=state.tls_id,
            time=state.time,
            agent=winner,
            urgency=claims[winner].urgency if winner in claims else 0.0,
            reason=reason,
            proposed=proposed,
            final=final,
            shield_note=note,
            explanation=agent.explain(),
            claims=claims_text(claims, self.rank),
        ))

    def _finish(self, started: float, state: JunctionState, allowed: list[bool], d: Decision) -> Decision:
        """Add the decision time and the what-if column."""
        d.latency_ms = round(1000 * (clock.perf_counter() - started), 3)
        if self.shadow is not None:
            d.fixed_time_would = self.shadow.act(state, [], allowed)
        return d
