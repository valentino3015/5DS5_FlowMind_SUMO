"""Tests for the basic safety shield (green-time rules).

Owner: pedestrian agent owner (basic tests written by Adem).
"""
from dataclasses import replace
from pathlib import Path

import yaml

from flowmind.contracts import make_example_state
from flowmind.control import shield

CONFIG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())


def test_before_min_green_only_current_phase():
    s = replace(make_example_state(), time_in_phase=5.0)       # min_green is 10
    assert shield.allowed(s, CONFIG) == [False, False, True, False]


def test_during_transition_only_current_phase():
    s = replace(make_example_state(), in_transition=True)
    assert shield.allowed(s, CONFIG) == [False, False, True, False]


def test_normal_case_everything_allowed():
    s = make_example_state()                                    # 14 s of green
    assert shield.allowed(s, CONFIG) == [True, True, True, True]


def test_after_max_green_current_phase_blocked():
    s = replace(make_example_state(), time_in_phase=60.0)      # max_green is 60
    assert shield.allowed(s, CONFIG) == [True, True, False, True]


def test_single_phase_is_never_blocked():
    s = replace(make_example_state(), num_phases=1, phase=0, time_in_phase=100.0,
                phase_lanes=(("north_in_0",),), phase_crossings=((),))
    assert shield.allowed(s, CONFIG) == [True]


def test_check_replaces_blocked_action():
    s = make_example_state()
    action, note = shield.check(s, 0, [False, True, True, True])
    assert action == 2 and note                                 # keeps the current phase, explains why
    action, note = shield.check(s, 1, [False, True, True, True])
    assert action == 1 and note == ""
