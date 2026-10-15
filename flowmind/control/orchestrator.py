"""Orchestrator (the manager): picks which agent controls a junction, every step.

Owner: Adem.

One Orchestrator per junction. Each step, decide():
1. asks every agent wants_control(state, events)
2. picks the claiming agent highest in orchestrator.priority (urgency breaks ties)
3. min_hold: an agent that just got control keeps it for orchestrator.min_hold
   seconds, unless a HIGHER-priority agent claims (an ambulance never waits)
4. if nobody claims, the fallback agent (max-pressure) drives the junction
5. the chosen agent's act() proposes a phase; the shield checks it
"""
from __future__ import annotations

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


class Orchestrator:
    def __init__(self, agents: list, priority: list[str], min_hold: float, fallback):
        """agents: Specialist objects; priority: agent names, most important first."""
        self.agents = {agent.name: agent for agent in agents}
        self.priority = priority
        self.min_hold = min_hold
        self.fallback = fallback
        self.current = fallback.name  # agent in control now
        self.since = 0.0              # time it got control

    def rank(self, name: str) -> int:
        """0 = most important. The fallback (not in the list) is the least important."""
        return self.priority.index(name) if name in self.priority else len(self.priority)

    def decide(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> Decision:
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
        return Decision(
            tls_id=state.tls_id,
            time=state.time,
            agent=winner,
            urgency=claims[winner].urgency if winner in claims else 0.0,
            reason=reason,
            proposed=proposed,
            final=final,
            shield_note=note,
            explanation=agent.explain(),
        )
