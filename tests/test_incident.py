"""Tests for the incident agent. Owner: Donia."""
from dataclasses import replace
from pathlib import Path

import yaml

from flowmind.agents.incident import IncidentAgent
from flowmind.contracts import make_example_state

CONFIG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())


def blocked_sequence(steps=20, dt=5.0):
    """north_in_0 stays full and never drains, while phase 0 gives it green."""
    base = make_example_state()
    out = []
    for i in range(steps):
        lanes = tuple(
            replace(l, queue=9, vehicles=10, occupancy=0.95, waiting_time=90 + 5 * i)
            if l.lane_id == "north_in_0" else l
            for l in base.lanes
        )
        out.append(replace(base, time=1200.0 + i * dt, phase=0, lanes=lanes))
    return out


def test_detects_blocked_lane():
    agent = IncidentAgent(CONFIG["incident"])
    claim = None
    for state in blocked_sequence():
        claim = agent.wants_control(state, []) or claim
    assert claim is not None
    assert "north_in_0" in claim.reason


def test_never_serves_the_blocked_lane():
    agent = IncidentAgent(CONFIG["incident"])
    states = blocked_sequence()
    for state in states:
        agent.wants_control(state, [])
    last = states[-1]
    phase = agent.act(last, [], [True] * last.num_phases)
    assert "north_in_0" not in last.phase_lanes[phase]


def test_no_false_alarm_on_normal_traffic():
    agent = IncidentAgent(CONFIG["incident"])
    base = make_example_state()
    for i in range(20):
        assert agent.wants_control(replace(base, time=1200.0 + 5 * i), []) is None