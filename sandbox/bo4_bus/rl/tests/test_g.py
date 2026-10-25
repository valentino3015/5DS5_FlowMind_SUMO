"""Agent G: action mask vs shield, reward regular_g terms, checkpoint rule, naming."""
import itertools
import json
import os
import sys

import numpy as np
import pytest

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
import reward as rw  # noqa: E402
from env import action_mask, shield  # noqa: E402
from test_obs_reward import b1_lanes, bus, lane, state  # noqa: E402

EPISODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "b1_mask_episode.json")


# ------------------------------------------------------------------ mask == shield
def agrees(current, tip, in_transition=False):
    m = action_mask(current, tip, in_transition)
    for a in (0, 1):
        assert m[a] == (shield(a, current, tip)[1] is None), (current, tip, a, m)
    return m


def test_mask_agrees_with_shield_on_a_recorded_episode():
    with open(EPISODE) as f:
        dec = json.load(f)["decisions"]
    rules = {d["rule"] for d in dec}
    assert {None, "min_green", "max_green"} <= rules           # every case is exercised
    for d in dec:
        m = agrees(d["current"], d["time_in_phase"])
        assert list(m) == d["mask"]                           # what the env exposed
        assert (d["rule"] is None) == bool(m[d["proposed"]])  # what the shield then did


def test_mask_agrees_with_shield_on_a_grid_and_is_never_empty():
    for cur, tip in itertools.product((0, 1), np.arange(0, 75, 0.5)):
        m = agrees(cur, float(tip))
        assert m.any()
    assert list(action_mask(0, 5.0)) == [True, False]          # min green: keep only
    assert list(action_mask(0, 30.0)) == [True, True]
    assert list(action_mask(1, 60.0)) == [True, False]         # max green: switch only
    assert list(action_mask(1, 30.0, in_transition=True)) == [False, True]


# ------------------------------------------------------------------ reward regular_g
def g_state(buses, green=0, halting=2):
    return state(b1_lanes(halting=halting), buses, green=green)


EARLY_W = bus("L1_EB_01", "A1B1_0", 60.0, late=-90.0, occ=40, speed=8.0)   # W, early, moving
EARLY_S = bus("L2_NB_01", "B0B1_0", 60.0, late=-90.0, occ=40, speed=8.0)   # S, early
LATE_N = bus("L2_SB_01", "B2B1_0", 80.0, late=30.0, occ=20, speed=8.0)     # N, late


def test_regular_g_without_new_terms_equals_f():
    s = g_state([EARLY_W, LATE_N])
    r_f, _ = rw.reward_regular(None, s, w=5.0)
    r_g, t = rw.reward_regular_g(s, s, w=5.0, w_early=0.0, c_switch=0.0)
    assert r_g == pytest.approx(r_f) and t["r_early"] == 0 and t["r_switch"] == 0


def test_early_penalty_for_green_to_an_early_bus():
    s = g_state([EARLY_W], green=0)                              # E-W green serves W
    r_f, _ = rw.reward_regular(None, s, w=5.0)
    r, t = rw.reward_regular_g(s, s, w=5.0, w_early=2.0, c_switch=0.0)
    early = 90.0 / 180.0 * 40                                    # earliness in H x persons
    assert t["early_person_headways"] == pytest.approx(early)
    assert t["r_early"] == pytest.approx(-2.0 * early * 5.0)
    assert r == pytest.approx(r_f - 2.0 * early * 5.0)
    # weight scales only this term
    r1, t1 = rw.reward_regular_g(s, s, w=5.0, w_early=1.0, c_switch=0.0)
    assert t["r_early"] == pytest.approx(2 * t1["r_early"])


def test_no_early_penalty_when_red_late_waiting_dwelling_or_far():
    assert rw.reward_regular_g(None, g_state([EARLY_S], green=0), w_early=3.0)[1]["r_early"] == 0
    s = g_state([EARLY_W, LATE_N], green=0)                      # a late bus waits (N, red)
    r, t = rw.reward_regular_g(None, s, w_early=3.0)
    assert t["r_early"] == 0 and t["late_bus_waiting"] == 1.0
    dwelling = bus("L1_EB_01", "A1B1_0", 60.0, late=-90.0, occ=40, at_stop=True)
    far = bus("L1_EB_01", "A1B1_0", 200.0, late=-90.0, occ=40, speed=8.0)
    unknown = bus("L1_EB_01", "A1B1_0", 60.0, late=None, occ=40, speed=8.0)
    for b in (dwelling, far, unknown):
        assert rw.reward_regular_g(None, g_state([b], green=0), w_early=3.0)[1]["r_early"] == 0
    # a late bus beyond 150 m does not block the penalty
    late_far = bus("L2_SB_01", "B2B1_0", 220.0, late=30.0, occ=20, speed=8.0)
    assert rw.reward_regular_g(None, g_state([EARLY_W, late_far]), w_early=1.0)[1]["r_early"] < 0


def test_switch_cost():
    prev, now = g_state([], green=0), g_state([], green=1)
    base, _ = rw.reward_regular_g(prev, prev, w_early=0.0, c_switch=2.0)
    r, t = rw.reward_regular_g(prev, now, w_early=0.0, c_switch=2.0)
    assert t["switched"] == 1.0 and t["r_switch"] == pytest.approx(-2.0 * rw.SWITCH_UNIT)
    assert r == pytest.approx(rw.reward_regular(None, now, w=5.0)[0] - 200.0)
    assert rw.reward_regular_g(prev, prev, c_switch=2.0)[1]["r_switch"] == 0
    assert rw.reward_regular_g(None, now, c_switch=2.0)[1]["r_switch"] == 0
    assert base == pytest.approx(rw.reward_regular(None, prev, w=5.0)[0])


def test_f_unchanged_and_g_registered():
    s = state([lane("A1B1_0", halting=3)], [bus("L1_EB_01", "A1B1_0", 20.0, late=90.0, occ=50)])
    assert rw.reward_regular(None, s, w=5.0)[0] == pytest.approx(
        rw.reward_person(None, s)[0] - 5.0 * 0.5 * 50 * 5.0)
    assert rw.REWARDS["regular_g"] is rw.reward_regular_g


# ------------------------------------------------------------------ checkpoints / naming
C_REF = {"never_inserted": 0.2, "person_delay_h": 250.0}


def val(bunch, pd_all9, ni=0.0, viol=0.0, viol_b1=0.0):
    return {"rl_violations": viol_b1, "all9_rl_violations": viol, "all9_never_inserted": ni,
            "all9_person_delay_h": pd_all9, "all9_bunch_last_total": bunch}


def test_better_stable_rule():
    from train_ppo import better_stable
    assert better_stable(val(30, 240), None, C_REF)
    assert not better_stable(val(20, 251), None, C_REF)          # worse person delay
    assert not better_stable(val(20, 240, ni=1.3), None, C_REF)  # > C_all's + 1
    assert better_stable(val(20, 240, ni=1.2), None, C_REF)
    assert not better_stable(val(20, 240, viol=1), None, C_REF)
    assert not better_stable(val(20, 240, viol_b1=1), None, C_REF)
    best = val(25, 245)
    assert better_stable(val(24, 249), best, C_REF)
    assert better_stable(val(25, 244), best, C_REF)              # tie -> lower person delay
    assert not better_stable(val(26, 200), best, C_REF)


def test_run_name_g():
    import train_ppo as tp
    assert tp.run_name("regular_g", 5, "v2", 0, 200000, 1, 0.5, True) == \
        "ppo_G_w5_e1_c0.5_v2_s0_200k"
    assert tp.run_name("regular_g", 5, "v2", 1, 200000, 3, 2, False) == \
        "ppo_G_w5_e3_c2_nomask_v2_s1_200k"
    assert tp.run_name("regular", 5, "v2", 0, 200000) == "ppo_F_w5_v2_s0_200k"   # F unchanged


def test_f_models_still_load_as_plain_ppo():
    import evaluate as ev
    path = os.path.join(RL, "runs", "ppo_F_w5_v2_s0_200k", "best_regular.zip")
    if not os.path.exists(path):
        pytest.skip("F model not present (runs/ is gitignored)")
    assert not ev.is_maskable(path)
    policy, version = ev.sb3_policy(path)
    assert version == "v2"
    obs = np.zeros(36, dtype=np.float32)
    assert policy(obs) == policy(obs, np.array([True, False]))   # PPO ignores the mask
