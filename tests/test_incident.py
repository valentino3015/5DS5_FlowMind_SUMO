"""Tests for the incident agent. Owner: Donia."""
from dataclasses import replace
from pathlib import Path

import yaml

from flowmind.agents.incident import IncidentAgent
from flowmind.contracts import make_example_state

CONFIG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())


def lane_sequence(queues, vehicles, occupancy=0.72, dt=5.0, lane_ids=("north_in_0",)):
    """lane_ids (default north_in_0) with the given queue and vehicle counts, while phase 0 gives them green.

    0.72 is what SUMO reports for a fully jammed lane (it counts car length, not gaps).
    """
    base = make_example_state()
    out = []
    for i, (q, v) in enumerate(zip(queues, vehicles)):
        lanes = tuple(
            replace(l, queue=q, vehicles=v, occupancy=occupancy, waiting_time=90 + 5 * i)
            if l.lane_id in lane_ids else l
            for l in base.lanes
        )
        out.append(replace(base, time=1200.0 + i * dt, phase=0, lanes=lanes))
    return out


def blocked_sequence(steps=20, lane_ids=("north_in_0",)):
    """lane_ids stay full and no vehicle leaves them."""
    return lane_sequence([9] * steps, [10] * steps, lane_ids=lane_ids)


def blocked_agent(lane_ids=("north_in_0",), model=None):
    """An agent that has detected lane_ids as blocked, and the last state it saw."""
    agent = IncidentAgent(CONFIG["incident"], model=model)
    states = blocked_sequence(lane_ids=lane_ids)
    for state in states:
        agent.wants_control(state, [])
    assert set(lane_ids) <= set(agent.blocked)
    return agent, states[-1]


class FakeModel:
    """Stands in for a trained PPO: always proposes the same phase."""

    def __init__(self, phase):
        self.phase = phase

    def predict(self, obs, deterministic=True):
        return self.phase, None


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


def test_detects_short_lane_where_every_car_is_stuck():
    # Seen in SUMO on a short 83 m lane: 7 stuck cars, occupancy only 0.36.
    assert first_claim(lane_sequence([7] * 20, [7] * 20, occupancy=0.36)) is not None


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


def test_never_wastes_green_on_a_fully_blocked_phase():
    # Phase 0 serves only north_in_0 and south_in_0: both blocked.
    agent, last = blocked_agent(("north_in_0", "south_in_0"))
    assert agent.act(last, [], [True] * last.num_phases) != 0


def test_mixed_phase_is_scored_without_its_blocked_lane():
    # Phase 0 still serves south_in_0, but nobody waits there: the 9 cars
    # stuck on north_in_0 must not count, so phase 1 (4 cars) wins.
    agent, last = blocked_agent()
    lanes = tuple(replace(l, queue=0) if l.lane_id == "south_in_0" else l for l in last.lanes)
    last = replace(last, lanes=lanes, time_in_phase=30.0)
    assert agent.act(last, [], [True] * last.num_phases) == 1


def test_ppo_choice_is_used_when_safe():
    agent, last = blocked_agent(("north_in_0", "south_in_0"), model=FakeModel(2))
    assert agent.act(last, [], [True] * last.num_phases) == 2
    assert agent.explain().startswith("PPO response")


def test_rule_replaces_an_unsafe_ppo_choice():
    agent, last = blocked_agent(("north_in_0", "south_in_0"), model=FakeModel(0))
    assert agent.act(last, [], [True] * last.num_phases) != 0
    assert agent.explain().startswith("PPO chose unsafe")


def test_observation_has_a_fixed_size():
    agent, last = blocked_agent()
    assert agent.encode(last).shape == (6 * CONFIG["incident"]["max_phases"] + 1,)


def test_no_false_alarm_on_normal_traffic():
    agent = IncidentAgent(CONFIG["incident"])
    base = make_example_state()
    for i in range(20):
        assert agent.wants_control(replace(base, time=1200.0 + 5 * i), []) is None

def test_event_claim_reason_stays_short():
    # The reason must not copy the event: the coordinator used to pass it on, and it grew every step.
    from flowmind.contracts import Event
    agent = IncidentAgent(CONFIG["incident"])
    e = Event(kind="incident", target_tls="J1", expires_at=2000.0,
              data={"cause": "spillback", "from_tls": "J2", "occupancy": 0.41, "reason": "x" * 500})
    claim = agent.wants_control(make_example_state(), [e])
    assert claim.reason == "spillback warning from J2, 41% full"
