"""Tests for the incident agent. Owner: Donia."""
from dataclasses import replace
from pathlib import Path

import yaml

from flowmind.agents.incident import IncidentAgent
from flowmind.contracts import make_example_state

CONFIG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())


def lane_sequence(queues, vehicles, occupancy=0.72, dt=5.0):
    """north_in_0 with the given queue and vehicle counts, while phase 0 gives it green.

    0.72 is what SUMO reports for a fully jammed lane (it counts car length, not gaps).
    """
    base = make_example_state()
    out = []
    for i, (q, v) in enumerate(zip(queues, vehicles)):
        lanes = tuple(
            replace(l, queue=q, vehicles=v, occupancy=occupancy, waiting_time=90 + 5 * i)
            if l.lane_id == "north_in_0" else l
            for l in base.lanes
        )
        out.append(replace(base, time=1200.0 + i * dt, phase=0, lanes=lanes))
    return out


def blocked_sequence(steps=20):
    """north_in_0 stays full and no vehicle leaves it."""
    return lane_sequence([9] * steps, [10] * steps)


def first_claim(states):
    agent = IncidentAgent(CONFIG["incident"])
    claim = None
    for state in states:
        claim = agent.wants_control(state, []) or claim
    return claim


def test_detects_blocked_lane():
    agent = IncidentAgent(CONFIG["incident"])
    claim = None
    for state in blocked_sequence():
        claim = agent.wants_control(state, []) or claim
    assert claim is not None
    assert "north_in_0" in claim.reason


def test_creeping_cars_do_not_hide_the_blockage():
    # Seen in SUMO: at green the halted count dips (cars creep forward)
    # but nobody leaves the lane, so the vehicle count does not drop.
    queues = [9, 6, 5, 9] * 5
    assert first_claim(lane_sequence(queues, [10] * 20)) is not None


def test_no_false_alarm_on_full_lane_that_drains():
    # A long queue at green that really leaves: vehicle count goes down.
    vehicles = [12, 9, 6, 12] * 5
    assert first_claim(lane_sequence([9] * 20, vehicles)) is None


def test_lets_go_to_recheck_then_releases_when_lane_moves():
    agent = IncidentAgent(CONFIG["incident"])
    states = blocked_sequence()
    for state in states:
        agent.wants_control(state, [])
    # The agent drives: the blocked lane gets no green (phase 2 serves east-west).
    last = states[-1]
    t = last.time
    for i in range(1, 30):
        claim = agent.wants_control(replace(last, time=t + 5 * i, phase=2), [])
    assert claim is None  # no confirmation for too long: let go to recheck
    # The lane gets green again and cars leave: blockage released.
    moving = lane_sequence([9] * 6, [10, 8, 6, 4, 3, 2])
    for s in moving:
        claim = agent.wants_control(replace(s, time=s.time + 1000), [])
    assert claim is None and not agent.blocked


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