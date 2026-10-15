import os
import sys

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
from env import shield  # noqa: E402


def test_keep_and_switch_allowed_inside_limits():
    assert shield(0, 0, 30.0) == (0, None, None)
    assert shield(1, 0, 10.0) == (1, None, None)


def test_min_green_blocks_early_switch():
    applied, rule, reason = shield(1, 0, 5.0)
    assert (applied, rule) == (0, "min_green") and "5 s of 10 s" in reason


def test_max_green_forces_switch_so_green_never_exceeds_60():
    assert shield(0, 0, 55.0) == (0, None, None)          # keep -> 60 s at the next tick
    applied, rule, _ = shield(0, 0, 60.0)
    assert (applied, rule) == (1, "max_green")
    assert shield(1, 1, 60.0)[0] == 0


def test_invalid_action_keeps():
    assert shield(5, 1, 30.0)[:2] == (1, "invalid")
