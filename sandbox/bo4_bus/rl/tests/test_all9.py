"""Per-junction geometry / observation (any of the 9 junctions) and the all9 scope.

B1 regression: tests/data/b1_golden.pkl holds 60 real JunctionStates of B1 (seed 900,
600 s warm-up + 1800 s, B1 switching every 20 s) with the v1 / v2 observations and the
B1 geometry computed by the code before the per-junction change (commit d34b558)."""
import os
import pickle
import sys

import numpy as np
import pytest

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
import observation as ob  # noqa: E402
from sense import APPROACHES, COMPASS, NEIGHBOURS, Geometry, compass_of  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "b1_golden.pkl")


@pytest.fixture(scope="module")
def net():
    import sumolib
    sys.path.insert(0, os.path.join(RL, "..", "nets", "grid"))
    from build_grid import NET
    return sumolib.net.readNet(NET, withInternal=True)


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def tls_ids():
    from build_grid import TLS_IDS
    return TLS_IDS


# ------------------------------------------------------------------ B1 unchanged
def test_b1_geometry_identical_to_before(net, golden):
    g = Geometry(net, "B1")
    assert g.approaches == APPROACHES == golden["approaches"]
    assert g.lanes_in == golden["lanes_in"] and g.lanes_out == golden["lanes_out"]
    assert g.neighbours == NEIGHBOURS
    assert g.straight == {a: ob.straight_out(a) for a in APPROACHES}
    assert Geometry(net).approaches == APPROACHES             # default tls is B1


def test_b1_observation_identical_to_before(net, golden):
    g = Geometry(net, "B1")
    assert len(golden["states"]) == 60 and sum(bool(s.approaching) for s in golden["states"]) > 30
    for s, v1, v2 in zip(golden["states"], golden["v1"], golden["v2"]):
        assert np.array_equal(ob.observe(s), v1)
        assert np.array_equal(ob.observe_at(s, g, "v1"), v1)
        assert np.array_equal(ob.observe(s, version="v2"), v2)
        assert np.array_equal(ob.observe_at(s, g, "v2"), v2)


# ------------------------------------------------------------------ all 9 junctions
def test_every_junction_has_four_approaches_in_compass_order(net, tls_ids):
    for tls in tls_ids:
        g = Geometry(net, tls)
        assert None not in g.approaches and len(set(g.approaches)) == 4, tls
        for side, a in zip(COMPASS, g.approaches):
            assert compass_of(net, net.getEdge(a).getFromNode().getID(), tls) == side
            assert net.getEdge(a + ".250.00").getToNode().getID() == tls
        assert len(g.lanes_in) == 20 and len(g.lanes_out) == 8, tls
        # straight exit leaves on the opposite side
        opposite = {"W": "E", "E": "W", "S": "N", "N": "S"}
        for side, a in zip(COMPASS, g.approaches):
            exit_to = net.getEdge(g.straight[a]).getToNode().getID()
            assert compass_of(net, exit_to, tls) == opposite[side], (tls, a)


def test_fringe_and_neighbour_details(net):
    a0 = Geometry(net, "A0")
    assert a0.approaches == ("left0A0", "B0A0", "bottom0A0", "A1A0")
    assert a0.straight["left0A0"] == "A0B0" and a0.neighbours == ("B0", "A1")
    a2 = Geometry(net, "A2")
    assert a2.approaches[3] == "top0A2"                       # fringe node north of A2
    assert Geometry(net, "C1").straight["B1C1"] == "C1right1"


def test_observation_v2_at_another_junction_uses_its_exits(net):
    from contract import JunctionState, LaneState
    g = Geometry(net, "A1")

    def lane(lid, occ=0.0, halting=0, length=34.0):
        return LaneState(lane_id=lid, length=length, vehicle_count=halting,
                         halting_count=halting, occupancy=occ, mean_speed=0.0,
                         waiting_time=0.0, co2_mg=0.0)
    incoming = [lane(lid, halting=3, length=ln) for lid, ln, _ in g.lanes_in]
    outgoing = [lane(lid, occ=0.8 if lid.startswith("A1B1") else 0.0, length=ln)
                for lid, ln in g.lanes_out]
    s = JunctionState(tls_id="A1", sim_time=0.0, green_phases=("a", "b"), current_green=0,
                      time_in_phase=0.0, in_transition=False, incoming=tuple(incoming),
                      outgoing=tuple(outgoing), approaching=())
    o = ob.observe_at(s, g, "v2")
    assert o.shape == (36,)
    assert all(o[5 * k] > 0 for k in range(4))                # every approach has a queue
    assert list(o[24 + 2::3]) == pytest.approx([0.8, 0.0, 0.0, 0.0])  # W -> A1B1 only
    # the B1 default would look for B1's edges and see nothing
    assert np.all(ob.observe(s)[:20] == 0)


def test_missing_approach_gives_zeros(net, golden):
    full = Geometry(net, "B1")
    no_w = Geometry(net, "B1", approaches=(None,) + APPROACHES[1:])
    assert all(not lid.startswith("A1B1") for lid, _, _ in no_w.lanes_in)
    assert "A1" not in no_w.neighbours
    for s in golden["states"]:
        a, b = ob.observe_at(s, full, "v2"), ob.observe_at(s, no_w, "v2")
        assert np.all(b[0:5] == 0) and np.all(b[24:27] == 0)
        keep = np.r_[5:24, 27:36]
        assert np.array_equal(a[keep], b[keep])


# ------------------------------------------------------------------ evaluate.py all9
def test_plan_and_output_names():
    import evaluate as ev
    m = {"F_s0": "a.zip", "F_s1": "b.zip"}
    assert ev.plan(m) == [("C_all", None, "b1"), ("F_s0", "a.zip", "b1"), ("F_s1", "b.zip", "b1")]
    assert [n for n, _, _ in ev.plan(m, "all9", with_b1=True)] == [
        "C_all", "F_s0@B1", "F_s0@all9", "F_s1@B1", "F_s1@all9"]
    assert [sc for _, _, sc in ev.plan(m, "all9")] == ["b1", "all9", "all9"]
    assert ev.experiment("final", "b1") == "rl_final" and ev.experiment("val", "b1") == "rl_val"
    assert ev.experiment("final", "all9") == "rl_all9"
    assert ev.experiment("val", "all9") == "rl_all9_val"


def test_junction_types_cover_the_grid_once(tls_ids):
    import evaluate as ev
    js = [j for t in ev.JUNCTION_TYPES.values() for j in t]
    assert sorted(js) == sorted(tls_ids)


def test_rl_controller_is_per_junction(net, tls_ids):
    import evaluate as ev
    ctrls = {t: ev.RLController(lambda o: 0, net, obs_version="v2", tls=t) for t in tls_ids}
    assert {c.signal.tls for c in ctrls.values()} == set(tls_ids)
    assert len({id(c.sensor) for c in ctrls.values()}) == 9      # own sensor / line memory
    assert len({id(c.signal) for c in ctrls.values()}) == 9      # own shield / monitor
    assert ctrls["B1"].geometry.approaches == APPROACHES


def test_paired_t():
    import evaluate as ev
    t, sig = ev.paired_t([1.0, 1.2, 0.9, 1.1, 1.0])
    assert t > 10 and sig
    t, sig = ev.paired_t([1.0, -1.0, 0.5, -0.5, 0.0])
    assert t == pytest.approx(0.0) and not sig
