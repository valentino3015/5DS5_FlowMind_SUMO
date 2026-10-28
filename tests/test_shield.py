"""Green-time tests for the safety shield, and the orchestrator's use of its allowed list.

Owner: pedestrian agent owner (basic tests written by Adem). The pedestrian rules
of the shield are tested in tests/test_pedestrian.py.
"""
from dataclasses import replace
from pathlib import Path

import yaml

from flowmind.contracts import make_example_state
from flowmind.control.orchestrator import shield_check
from flowmind.control.shield import SafetyShield

CONFIG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())
SHIELD = SafetyShield(CONFIG["signals"], CONFIG["shield"])


def quiet_state(**changes):
    """The example junction with nobody at the crossings, so only green-time rules apply."""
    return replace(make_example_state(), crossings=(), **changes)


def test_before_min_green_only_current_phase():
    s = quiet_state(time_in_phase=5.0)                          # min_green is 10
    assert SHIELD.allowed_phases(s) == [False, False, True, False]


def test_during_transition_only_current_phase():
    s = quiet_state(in_transition=True)
    assert SHIELD.allowed_phases(s) == [False, False, True, False]


def test_normal_case_everything_allowed():
    s = quiet_state()                                           # 14 s of green
    assert SHIELD.allowed_phases(s) == [True, True, True, True]


def test_after_max_green_current_phase_blocked():
    s = quiet_state(time_in_phase=60.0)                         # max_green is 60
    assert SHIELD.allowed_phases(s) == [True, True, False, True]


def test_single_phase_is_never_blocked():
    s = quiet_state(num_phases=1, phase=0, time_in_phase=100.0,
                    phase_lanes=(("north_in_0",),), phase_crossings=((),))
    assert SHIELD.allowed_phases(s) == [True]


def test_check_replaces_blocked_action():
    s = make_example_state()
    action, note = shield_check(s, 0, [False, True, True, True])
    assert action == 2 and note                                 # keeps the current phase, explains why
    action, note = shield_check(s, 1, [False, True, True, True])
    assert action == 1 and note == ""
