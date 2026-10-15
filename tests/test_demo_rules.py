"""Tests for Adem's temporary demo emergency rule.

Owner: Adem.
"""
from dataclasses import replace
from pathlib import Path

import yaml

from flowmind.agents.demo_rules import DemoEmergency
from flowmind.contracts import Event, SpecialVehicle, make_example_state

CONFIG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())
ALL = [True, True, True, True]


def with_ambulance(distance):
    amb = SpecialVehicle("amb_1", "emergency", "west_in_0", distance=distance, speed=12.0, delay=None)
    s = make_example_state()
    return replace(s, special_vehicles=s.special_vehicles + (amb,))


def test_no_ambulance_no_claim():
    agent = DemoEmergency(CONFIG["emergency"])
    s = make_example_state()                         # only a bus
    assert agent.wants_control(s, []) is None
    assert agent.act(s, [], ALL) == s.phase


def test_near_ambulance_gets_its_phase():
    agent = DemoEmergency(CONFIG["emergency"])
    s = replace(with_ambulance(80.0), phase=0)       # currently north-south
    claim = agent.wants_control(s, [])
    assert claim.urgency == 1.0 and "amb_1" in claim.reason
    assert agent.act(s, [], ALL) == 2                 # west_in_0 is served by phase 2
    assert "amb_1" in agent.explain()


def test_far_ambulance_is_ignored():
    agent = DemoEmergency(CONFIG["emergency"])
    assert agent.wants_control(with_ambulance(500.0), []) is None


def test_coordinator_warning_preclears_only_when_close():
    agent = DemoEmergency(CONFIG["emergency"])
    s = replace(make_example_state(), phase=0)
    soon = Event("emergency", "J1", 1300.0, {"vehicle_id": "amb_1", "eta": 15.0, "lane_id": "east_in_0"})
    later = Event("emergency", "J1", 1300.0, {"vehicle_id": "amb_1", "eta": 45.0, "lane_id": "east_in_0"})
    assert agent.wants_control(s, [later]) is None   # preclear_eta is 20 s
    assert agent.wants_control(s, [soon]) is not None
    assert agent.act(s, [soon], ALL) == 2


def test_blocked_phase_falls_back_to_allowed():
    agent = DemoEmergency(CONFIG["emergency"])
    s = with_ambulance(80.0)
    assert agent.act(s, [], [True, True, False, True]) == 0   # phase 2 blocked -> keep_phase picks 0
