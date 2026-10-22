"""Observation v2, reward F, and the run naming / model options (no SUMO)."""
import os
import sys

import numpy as np
import pytest

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
import observation as ob  # noqa: E402
import reward as rw  # noqa: E402
from contract import JunctionState, LaneState  # noqa: E402
from sense import APPROACHES  # noqa: E402
from test_obs_reward import b1_lanes, bus, lane  # noqa: E402


def out_lane(lid, occ):
    return LaneState(lane_id=lid, length=234.0, vehicle_count=0, halting_count=0,
                     occupancy=occ, mean_speed=0.0, waiting_time=0.0, co2_mg=0.0)


def b1_out(occ=None):
    """B1's 8 outgoing lanes; occ: {lane id: occupancy}."""
    occ = occ or {}
    return [out_lane(f"B1{e}_{i}", occ.get(f"B1{e}_{i}", 0.0))
            for e in ("A1", "B0", "B2", "C1") for i in range(2)]


def state(incoming=(), approaching=(), outgoing=(), green=0, tip=0.0, transition=False):
    return JunctionState(tls_id="B1", sim_time=0.0, green_phases=("GGrr", "rrGG"),
                         current_green=green, time_in_phase=tip, in_transition=transition,
                         incoming=tuple(incoming), outgoing=tuple(outgoing),
                         approaching=tuple(approaching))


# ------------------------------------------------------------------ observation v1 / v2
def test_v1_is_still_the_default_and_unchanged():
    s = state(b1_lanes(), [bus("L1_EB_01", "A1B1_0", 30.0, late=90.0)], b1_out({"B1C1_0": 0.9}))
    assert ob.observe(s).shape == (24,) == (ob.OBS_SIZE,)
    assert np.array_equal(ob.observe(s), ob.observe(s, version="v1"))
    assert ob.bounds("v1")[0] is ob.LOW and ob.bounds("v1")[1] is ob.HIGH


def test_v2_length_bounds_and_v1_prefix():
    s = state(b1_lanes(halting=4), [bus("L1_EB_01", "A1B1_0", 30.0, late=90.0, occ=45)],
              b1_out({"B1C1_0": 0.4}), green=1, tip=20.0)
    v1, v2 = ob.observe(s, version="v1"), ob.observe(s, version="v2")
    assert v2.shape == (36,) == (ob.OBS_SIZE_V2,) and v2.dtype == np.float32
    assert np.array_equal(v2[:24], v1)                      # same 24 values, same order
    lo, hi = ob.bounds("v2")
    assert lo.shape == hi.shape == (36,)
    assert list(hi[24:]) == [2.0, 1.0, 1.0] * 4 and np.all(lo[24:] == 0)
    assert ob.version_of_size(24) == "v1" and ob.version_of_size(36) == "v2"


def test_v2_empty_state_extra_values_are_zero():
    o = ob.observe(state(b1_lanes(), outgoing=b1_out()), version="v2")
    assert np.all(o[24:] == 0)


def test_straight_out_mapping():
    assert [ob.straight_out(a) for a in APPROACHES] == ["B1C1", "B1A1", "B1B2", "B1B0"]


def test_v2_values_per_approach_in_order():
    buses = [bus("L1_EB_03", "A1B1_0", 120.0, late=-40.0),             # W, 2nd bus
             bus("L1_EB_04", "A1B1.250.00_0", 30.0, late=90.0),        # W nearest
             bus("L1_WB_02", "C1B1_0", 75.0, late=None),               # E, lateness unknown
             bus("L2_NB_01", "B0B1_0", 100.0, late=400.0),             # S, very late
             bus("L2_NB_02", "B0B1_1", 200.0, late=0.0),               # S, beyond 150 m
             bus("L2_SB_01", "B2B1_0", 10.0, late=-300.0)]             # N, very early
    out = b1_out({"B1C1_0": 0.2, "B1C1_1": 0.7, "B1A1_1": 0.1, "B1B0_0": 1.0})
    o = ob.observe(state(b1_lanes(), buses, out), version="v2")
    w, e, s, n = o[24:27], o[27:30], o[30:33], o[33:36]
    assert list(w) == pytest.approx([(90 + 180) / 180, 1.0, 0.7])  # max of B1C1's lanes
    assert list(e) == pytest.approx([0.0, 0.0, 0.1])
    assert list(s) == pytest.approx([2.0, 0.0, 0.0])               # clipped at 2 H
    assert list(n) == pytest.approx([0.0, 0.0, 1.0])               # early -> clipped at 0
    assert o[3] == 0.5 and o[13] == 1.0                            # v1 bus_late unchanged


def test_v2_always_in_bounds():
    buses = [bus("L1_EB_01", "A1B1_0", 0.0, late=1e5, occ=500),
             bus("L1_EB_02", "A1B1_1", 5.0, late=-1e5),
             bus("L1_WB_01", "C1B1_0", 10.0, late=-1e5)]
    out = b1_out({lid: 3.0 for lid in ("B1C1_0", "B1A1_0", "B1B0_1", "B1B2_1")})
    o = ob.observe(state(b1_lanes(halting=500), buses, out, tip=500.0), version="v2")
    lo, hi = ob.bounds("v2")
    assert np.all(o >= lo) and np.all(o <= hi)


def test_unknown_version_rejected():
    with pytest.raises(ValueError):
        ob.observe(state(b1_lanes()), version="v3")


# ------------------------------------------------------------------ reward F
def f_state():
    lanes = [lane("A1B1_0", halting=6), lane("C1B1_0", halting=3), lane("B0B1_0", halting=1)]
    buses = [bus("L1_EB_01", "A1B1_0", 170.0, late=200.0, occ=40, at_stop=True),  # dwelling
             bus("L1_WB_01", "C1B1_0", 20.0, late=90.0, occ=50),                  # queued, late
             bus("L2_NB_01", "B0B1_0", 20.0, late=-60.0, occ=30),                 # queued, early
             bus("L2_SB_01", "C1B1_0", 60.0, late=None, occ=20),                  # queued, unknown
             bus("L2_SB_02", "B2B1_0", 90.0, late=300.0, occ=60, speed=8.0)]      # moving
    return state(lanes, buses)


def test_reward_f_is_e_minus_late_term():
    s = f_state()
    r_e, _ = rw.reward_person(None, s)
    r, terms = rw.reward_regular(None, s)
    late = 90.0 / 180.0 * 50                                  # only the queued late bus
    assert terms["late_person_headways"] == pytest.approx(late)
    assert terms["r_late"] == pytest.approx(-late * 5.0)
    assert terms["r_person"] == pytest.approx(r_e)
    assert r == pytest.approx(r_e - late * 5.0) and terms["r_regular"] == r


def test_reward_f_weight_scales_only_the_late_term():
    s = f_state()
    r_e, _ = rw.reward_person(None, s)
    r1, t1 = rw.reward_regular(None, s, w=1.0)
    r2, t2 = rw.reward_regular(None, s, w=2.0)
    r0, _ = rw.reward_regular(None, s, w=0.0)
    assert r0 == pytest.approx(r_e)
    assert t2["r_late"] == pytest.approx(2 * t1["r_late"])
    assert r2 - r_e == pytest.approx(2 * (r1 - r_e))
    assert rw.reward_regular(None, s, w=0.5)[0] == pytest.approx(r_e + 0.5 * t1["r_late"])


def test_reward_f_without_late_buses_equals_e_and_never_positive():
    s = state(b1_lanes(halting=2), [bus("L1_EB_01", "A1B1_0", 10.0, late=-20.0)])
    r_f, r_e = rw.reward_regular(None, s)[0], rw.reward_person(None, s)[0]
    assert r_f == pytest.approx(r_e) and r_f < 0
    assert rw.reward_regular(None, state(b1_lanes()))[0] == 0.0


def test_rewards_d_and_e_unchanged():
    """Same values as before reward F was added (see test_obs_reward)."""
    s = f_state()
    # 10 halting on the lanes - 4 halted buses = 6 cars; 3 queued buses (1 dwells)
    assert rw.reward_vehicle(None, s)[0] == -(6 + 3) * 5.0
    assert rw.reward_person(None, s)[0] == pytest.approx(-(1.3 * 6 + 50 + 30 + 20) * 5.0)
    assert rw.REWARDS["vehicle"] is rw.reward_vehicle and rw.REWARDS["person"] is rw.reward_person
    assert rw.REWARDS["regular"] is rw.reward_regular


# ------------------------------------------------------------------ options
def test_run_name():
    import train_ppo as tp
    assert tp.run_name("regular", 1.0, "v2", 0, 200000) == "ppo_F_w1_v2_s0_200k"
    assert tp.run_name("regular", 0.5, "v2", 3, 200000) == "ppo_F_w0.5_v2_s3_200k"
    assert tp.run_name("person", 1.0, "v1", 0, 5000) == "ppo_E_v1_s0_5k"


def test_parse_models(tmp_path):
    import evaluate as ev
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    a.write_bytes(b""), b.write_bytes(b"")
    m = ev.parse_models([f"RL_F={a}", f"RL_E={b}"])
    assert list(m) == ["RL_F", "RL_E"] and m["RL_F"] == os.path.abspath(a)
    for bad in ([f"RL_F={a}", f"RL_F={b}"], [f"C_all={a}"], [str(a)],
                [f"RL_F={tmp_path / 'missing.zip'}"]):
        with pytest.raises(ValueError):
            ev.parse_models(bad)


@pytest.mark.parametrize("run", ["ppo_D_200k", "ppo_E_200k"])
def test_trained_d_e_models_still_load_and_run_v1(run):
    path = os.path.join(RL, "runs", run, "best_model.zip")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present (runs/ is gitignored)")
    import evaluate as ev
    policy, version = ev.sb3_policy(path)
    assert version == "v1"
    s = state(b1_lanes(halting=3), [bus("L1_EB_01", "A1B1_0", 30.0, late=60.0)])
    assert policy(ob.observe(s, version=version)) in (0, 1)


# ------------------------------------------------------------------ F checkpoints
C_REF = {"person_delay_B1_h": 6.0, "bunch_last_total": 2.4}


def val(pd_b1, bunch):
    return {"person_delay_B1_h": pd_b1, "bunch_last_total": bunch, "veh_delay_B1": 20.0,
            "rl_violations": 0.0}


def test_better_regular_rule():
    from train_ppo import better_regular
    assert not better_regular(val(6.1, 0.0), None, C_REF)          # worse than C_all
    assert better_regular(val(6.0, 3.0), None, C_REF)              # equal to C_all qualifies
    best = val(5.0, 2.0)
    assert better_regular(val(5.9, 1.8), best, C_REF)              # less bunching
    assert not better_regular(val(4.0, 2.2), best, C_REF)          # more bunching
    assert better_regular(val(4.9, 2.0), best, C_REF)              # tie -> lower person delay
    assert not better_regular(val(5.0, 2.0), best, C_REF)          # identical: keep the first


class FakeLogger:
    def record(self, *a, **k):
        pass

    def dump(self, *a, **k):
        pass


class FakeModel:
    logger = FakeLogger()

    def __init__(self):
        self.saved = []

    def save(self, path):
        self.saved.append(os.path.basename(path))
        open(path, "wb").close()


def run_callback(tmp_path, monkeypatch, reward, results):
    """Drive the validation callback over (steps, metrics) without SUMO."""
    import json
    import train_ppo as tp
    seq = iter(results)
    monkeypatch.setattr(tp, "validate", lambda *a, **k: dict(next(seq)[1]))
    cb = tp.make_callback(str(tmp_path), reward, 1000, 600.0, "v2",
                          C_REF if reward == "regular" else None)
    model = FakeModel()
    cb.model = model
    for steps, _ in results:
        cb.num_timesteps = steps
        cb._evaluate()
    with open(tmp_path / "best.json") as f:
        return json.load(f), model.saved


def test_regular_keeps_best_model_and_best_regular(tmp_path, monkeypatch):
    results = [(0, val(7.0, 1.0)),          # best person delay so far, not within C_all
               (1000, val(5.5, 2.0)),       # best_model and first best_regular
               (2000, val(5.9, 1.6)),       # best_regular (less bunching, within C_all)
               (3000, val(5.0, 3.0)),       # best_model (lowest person delay), more bunching
               (4000, val(6.5, 0.2))]       # least bunching but worse than C_all
    best, saved = run_callback(tmp_path, monkeypatch, "regular", results)
    assert best["steps"] == 3000 and best["person_delay_B1_h"] == 5.0
    assert best["best_regular"]["steps"] == 2000
    assert best["best_regular"]["bunch_last_total"] == 1.6
    assert best["c_all_reference"] == C_REF
    assert [s for s in saved if s.startswith("best")] == [
        "best_model.zip", "best_model.zip", "best_regular.zip", "best_regular.zip",
        "best_model.zip"]
    summary = __import__("pandas").read_csv(tmp_path / "validation_summary.csv")
    assert list(summary.best_regular) == [False, True, True, False, False]


def test_regular_best_regular_none_until_a_checkpoint_qualifies(tmp_path, monkeypatch):
    best, saved = run_callback(tmp_path, monkeypatch, "regular", [(0, val(7.0, 1.0))])
    assert best["best_regular"] is None and "best_regular.zip" not in saved


def test_person_and_vehicle_unchanged(tmp_path, monkeypatch):
    results = [(0, val(7.0, 1.0)), (1000, val(5.5, 2.0)), (2000, val(5.9, 0.0))]
    best, saved = run_callback(tmp_path, monkeypatch, "person", results)
    assert best["steps"] == 1000 and set(best) == {"steps", *val(0, 0)}
    assert "best_regular.zip" not in saved
    summary = __import__("pandas").read_csv(tmp_path / "validation_summary.csv")
    assert "best_regular" not in summary
