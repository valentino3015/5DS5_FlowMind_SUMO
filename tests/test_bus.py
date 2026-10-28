"""Unit tests for the bus agent (BO4 conditional priority). Owner: Firas.

Run with: pytest tests/test_bus.py
"""
from dataclasses import replace

from flowmind.agents.bus import BusAgent, line_of
from flowmind.contracts import LaneState, SpecialVehicle, make_example_state

PARAMS = {"detect_distance": 150, "headway": 300, "late_threshold": 0,
          "max_queue": 15, "max_hold": 20, "urgency_scale": 300}
ALL = [True, True, True, True]


def state_at(t, buses, lanes=None, phase=2):
    s = make_example_state()
    return replace(s, time=float(t), special_vehicles=tuple(buses), phase=phase,
                   lanes=lanes if lanes is not None else s.lanes)


def bus(vid, dist, lane="north_in_0"):
    return SpecialVehicle(vid, "bus", lane, distance=float(dist), speed=8.0, delay=None)


def agent_with_passage(t_pass=0.0, line="bus_ns"):
    """An agent that saw a bus of `line` cross at t_pass."""
    a = BusAgent(PARAMS)
    a.wants_control(state_at(t_pass - 5, [bus(f"{line}.0", 10)]), [])
    a.wants_control(state_at(t_pass, []), [])   # bus gone close to the line -> crossed
    return a


def test_line_of():
    assert line_of("bus_we.3") == "bus_we"
    assert line_of("L1_EB_09") == "L1_EB"
    assert line_of("bus") == "bus"


def test_no_previous_bus_no_claim():
    a = BusAgent(PARAMS)
    assert a.wants_control(state_at(100, [bus("bus_ns.1", 100)]), []) is None
    assert "no previous bus" in a.explain()


def test_passage_is_recorded():
    a = agent_with_passage(0.0)
    assert a.last_passage["bus_ns"] == 0.0


def test_late_bus_claims_with_urgency():
    a = agent_with_passage(0.0)
    claim = a.wants_control(state_at(400, [bus("bus_ns.1", 100)]), [])   # 100 s late
    assert claim is not None
    assert abs(claim.urgency - 100 / 300) < 1e-9
    assert "100 s late" in claim.reason


def test_early_bus_is_not_helped():
    a = agent_with_passage(0.0)
    assert a.wants_control(state_at(200, [bus("bus_ns.1", 100)]), []) is None   # 100 s early
    assert "early" in a.explain()


def test_far_bus_is_ignored():
    a = agent_with_passage(0.0)
    assert a.wants_control(state_at(400, [bus("bus_ns.1", 400)]), []) is None


def test_queue_rule_refuses():
    a = agent_with_passage(0.0)
    s = make_example_state()
    lanes = tuple(replace(l, queue=30) if l.lane_id == "east_in_0" else l for l in s.lanes)
    assert a.wants_control(state_at(400, [bus("bus_ns.1", 100)], lanes=lanes), []) is None
    assert "cross queue" in a.explain()


def test_act_recalls_green_for_bus_lane():
    a = agent_with_passage(0.0)
    s = state_at(400, [bus("bus_ns.1", 100)], phase=2)       # E-W green, bus on north_in_0
    assert a.wants_control(s, []) is not None
    assert a.act(s, [], ALL) == 0                             # N-S straight serves north_in_0
    assert "early recall" in a.explain()


def test_act_extends_when_already_green():
    a = agent_with_passage(0.0)
    s = state_at(400, [bus("bus_ns.1", 100)], phase=0)
    a.wants_control(s, [])
    assert a.act(s, [], ALL) == 0
    assert "extend green" in a.explain()


def test_act_respects_shield():
    a = agent_with_passage(0.0)
    s = state_at(400, [bus("bus_ns.1", 100)], phase=2)
    a.wants_control(s, [])
    assert a.act(s, [], [False, False, True, False]) == 2     # only current allowed: keep


def test_budget_stops_priority():
    a = agent_with_passage(0.0)
    for t in range(400, 430, 5):                              # served 25 s > max_hold 20
        claim = a.wants_control(state_at(t, [bus("bus_ns.1", 100)]), [])
    assert claim is None
    assert "budget" in a.explain()


def test_latest_bus_wins():
    a = agent_with_passage(0.0, "bus_ns")
    a.wants_control(state_at(45, [bus("bus_we.0", 10, "east_in_0")]), [])
    a.wants_control(state_at(50, []), [])                     # bus_we crossed at 50
    claim = a.wants_control(state_at(400, [bus("bus_ns.1", 100), bus("bus_we.1", 60, "east_in_0")]), [])
    assert claim is not None and "bus_ns.1" in claim.reason  # 100 s late beats 50 s late


def test_fast_bus_passage_is_recorded():
    # A bus last seen 70 m away (one 5 s step at 13.9 m/s) and then gone has still crossed,
    # so the bus right behind it is early (bunched), not late.
    a = BusAgent(PARAMS)
    a.last_passage["bus_ns"] = -500.0
    a.wants_control(state_at(95, [bus("bus_ns.0", 70)]), [])
    a.wants_control(state_at(100, []), [])
    assert a.last_passage["bus_ns"] == 100.0
    assert a.wants_control(state_at(110, [bus("bus_ns.1", 100)]), []) is None
