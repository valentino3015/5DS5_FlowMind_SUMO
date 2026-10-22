"""Tests for the shared contracts and the agent stubs.

Owner: Adem. Run with: pytest
"""
from pathlib import Path

import pytest
import yaml

from flowmind.agents.baselines import FixedTime, MaxPressure
from flowmind.agents.bus import BusAgent
from flowmind.agents.emergency import EmergencyAgent
from flowmind.agents.flow import FlowAgent
from flowmind.agents.incident import IncidentAgent
from flowmind.agents.pedestrian import PedestrianAgent
from flowmind.contracts import Claim, JunctionState, example_phase_movements, make_example_state

CONFIG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())


def test_example_state_is_valid():
    s = make_example_state()
    assert isinstance(s, JunctionState)
    assert 0 <= s.phase < s.num_phases
    assert s.time_in_phase >= 0
    assert isinstance(s.lanes, tuple) and isinstance(s.special_vehicles, tuple)
    for lane in s.lanes + s.out_lanes:
        assert 0 <= lane.queue <= lane.vehicles
        assert 0.0 <= lane.occupancy <= 1.0
    for v in s.special_vehicles:
        assert v.kind in ("bus", "emergency")
        assert v.distance >= 0
    for c in s.crossings:
        assert c.waiting >= 0 and c.max_wait >= 0
    # every movement uses lanes that exist in the example
    lane_ids = {lane.lane_id for lane in s.lanes + s.out_lanes}
    for moves in example_phase_movements().values():
        for src, dst in moves:
            assert src in lane_ids and dst in lane_ids
    # one entry per green phase, consistent with example_phase_movements()
    assert len(s.phase_lanes) == len(s.phase_crossings) == s.num_phases
    for phase, moves in example_phase_movements().items():
        assert set(s.phase_lanes[phase]) == {src for src, _ in moves}
    crossing_ids = {c.crossing_id for c in s.crossings}
    for ids in s.phase_crossings:
        assert set(ids) <= crossing_ids


def test_example_state_is_frozen():
    s = make_example_state()
    with pytest.raises(Exception):
        s.phase = 0


def make_agents():
    return [
        FixedTime(CONFIG["fixed_time"]),
        MaxPressure(CONFIG["max_pressure"], example_phase_movements()),
        FlowAgent(CONFIG["flow"]),
        EmergencyAgent(CONFIG["emergency"]),
        BusAgent(CONFIG["bus"]),
        PedestrianAgent(CONFIG["pedestrian"]),
        IncidentAgent(CONFIG["incident"]),
    ]


@pytest.mark.parametrize("allowed", [
    [True, True, True, True],     # everything allowed
    [True, False, False, True],   # current phase (2) blocked by the shield
])
def test_agents_return_an_allowed_phase(allowed):
    s = make_example_state()
    for agent in make_agents():
        claim = agent.wants_control(s, [])
        assert claim is None or (isinstance(claim, Claim) and 0.0 <= claim.urgency <= 1.0)
        action = agent.act(s, [], allowed)
        assert isinstance(action, int), agent.name
        assert allowed[action], f"{agent.name} returned blocked phase {action}"
        assert isinstance(agent.explain(), str)


def test_max_pressure_picks_congested_north_south():
    s = make_example_state()
    agent = MaxPressure({}, example_phase_movements())
    assert agent.act(s, [], [True] * 4) == 0   # north-south straight has the longest queues


def test_phase_for_lane():
    s = make_example_state()                     # current phase is 2
    agent = BusAgent({})
    all_ok = [True] * 4
    assert agent.phase_for_lane(s, "north_in_0", all_ok) == 0
    assert agent.phase_for_lane(s, "east_in_0", all_ok) == 2      # already green: keep it
    assert agent.phase_for_lane(s, "west_in_1", all_ok) == 3
    assert agent.phase_for_lane(s, "north_in_0", [False, True, True, True]) is None   # blocked
    assert agent.phase_for_lane(s, "no_such_lane", all_ok) is None


def test_phase_for_crossing():
    s = make_example_state()
    agent = PedestrianAgent({})
    all_ok = [True] * 4
    assert agent.phase_for_crossing(s, ":J1_c0", all_ok) == 0
    assert agent.phase_for_crossing(s, ":J1_c1", all_ok) == 2
    assert agent.phase_for_crossing(s, ":J1_c0", [False, True, True, True]) is None   # blocked
    assert agent.phase_for_crossing(s, "no_such_crossing", all_ok) is None
