"""Tests for the static green wave baseline (core idea 6, safe version).

Owner: Adem.
"""
from dataclasses import replace

from flowmind.agents.baselines import GreenWave, green_wave_offset
from flowmind.contracts import make_example_state

PARAMS = {"green": 30, "link_length": 200, "speed": 13.89}
ORDER = ["J1", "J2", "J3"]
ALL = [True, True, True, True]


def test_offsets_follow_travel_time():
    assert green_wave_offset(PARAMS, ORDER, "J1") == 0
    assert round(green_wave_offset(PARAMS, ORDER, "J2"), 1) == 14.4
    assert round(green_wave_offset(PARAMS, ORDER, "J3"), 1) == 28.8


def test_downstream_junction_switches_later():
    j1 = GreenWave(PARAMS, offset=0.0)
    j2 = GreenWave(PARAMS, offset=14.4)
    s = replace(make_example_state(), time=35.0)     # 4 phases of 30 s
    assert j1.act(s, [], ALL) == 1                    # J1 already moved on to phase 1
    assert j2.act(s, [], ALL) == 0                    # J2 still in phase 0 (14.4 s later)
    assert j2.act(replace(s, time=45.0), [], ALL) == 1


def test_blocked_phase_keeps_an_allowed_one():
    j1 = GreenWave(PARAMS, offset=0.0)
    s = replace(make_example_state(), time=35.0)     # schedule says phase 1, current phase 2
    assert j1.act(s, [], [True, False, True, True]) == 2
