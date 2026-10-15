"""Tests for the orchestrator, with tiny fake agents (no SUMO needed).

Owner: Adem.
"""
from dataclasses import replace

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, make_example_state
from flowmind.control.orchestrator import Orchestrator

PRIORITY = ["emergency", "incident", "pedestrian", "bus", "flow"]
ALL = [True, True, True, True]


class Fake(Specialist):
    """Claims when self.claim is set, always proposes self.phase."""

    def __init__(self, name, phase, claim=None):
        super().__init__()
        self.name, self.phase, self.claim = name, phase, claim

    def wants_control(self, state, events):
        return self.claim

    def act(self, state, events, allowed):
        self.last_reason = f"{self.name} wants phase {self.phase}"
        return self.phase


def make(bus_claim=None, ped_claim=None, emergency_claim=None):
    agents = [Fake("bus", 0, bus_claim), Fake("pedestrian", 1, ped_claim), Fake("emergency", 3, emergency_claim)]
    return Orchestrator(agents, PRIORITY, min_hold=10, fallback=Fake("max_pressure", 2)), agents


def at(time):
    return replace(make_example_state(), time=time)


def test_nobody_claims_uses_fallback():
    orch, _ = make()
    d = orch.decide(at(100), [], ALL)
    assert d.agent == "max_pressure" and d.final == 2 and d.urgency == 0.0


def test_priority_beats_urgency():
    orch, _ = make(bus_claim=Claim(0.9, "late bus"), ped_claim=Claim(0.2, "people waiting"))
    d = orch.decide(at(100), [], ALL)
    assert d.agent == "pedestrian" and d.final == 1 and d.reason == "people waiting"


def test_min_hold_keeps_agent_against_lower_priority():
    orch, agents = make(bus_claim=Claim(0.5, "late bus"))
    assert orch.decide(at(100), [], ALL).agent == "bus"
    agents[0].claim = None                         # bus passed: nobody claims, fallback would win
    d = orch.decide(at(105), [], ALL)
    assert d.agent == "bus" and "min_hold" in d.reason
    assert orch.decide(at(110), [], ALL).agent == "max_pressure"   # hold is over


def test_higher_priority_preempts_immediately():
    orch, agents = make(bus_claim=Claim(0.5, "late bus"))
    orch.decide(at(100), [], ALL)
    agents[2].claim = Claim(1.0, "ambulance")
    d = orch.decide(at(101), [], ALL)
    assert d.agent == "emergency" and d.final == 3


def test_shield_overrides_blocked_action():
    orch, _ = make(emergency_claim=Claim(1.0, "ambulance"))
    d = orch.decide(at(100), [], [True, True, True, False])        # phase 3 blocked
    assert d.proposed == 3 and d.final == 2 and d.shield_note
