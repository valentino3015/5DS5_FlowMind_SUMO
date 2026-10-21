"""Tests for the corridor coordinator (no SUMO needed).

Owner: Adem.
"""
from types import SimpleNamespace

from flowmind.control.coordinator import Coordinator

PARAMS = {"horizon": 60, "min_speed": 2, "margin": 5, "incident_ttl": 30}
ORDER = ["J1", "J2", "J3"]


def test_emergency_eta_and_horizon():
    c = Coordinator(PARAMS, ORDER)
    # ambulance at 10 m/s: J2 is 200 m away (20 s), J3 is 700 m away (70 s > horizon)
    approaches = [("amb_1", "J2", 200.0, 10.0, "J1_J2_1"), ("amb_1", "J3", 700.0, 10.0, "J2_J3_1")]
    events = c.update(100.0, approaches, [])
    assert events["J3"] == [] and events["J1"] == []
    (e,) = events["J2"]
    assert e.kind == "emergency" and e.data["eta"] == 20.0 and e.data["lane_id"] == "J1_J2_1"
    assert e.expires_at == 100.0 + 20.0 + 5


def test_stopped_ambulance_uses_min_speed():
    c = Coordinator(PARAMS, ORDER)
    (e,) = c.update(0.0, [("amb_1", "J1", 50.0, 0.0, "W_J1_1")], [])["J1"]
    assert e.data["eta"] == 25.0          # 50 m / 2 m/s


def test_warning_disappears_when_ambulance_is_gone():
    c = Coordinator(PARAMS, ORDER)
    assert c.update(0.0, [("amb_1", "J1", 50.0, 10.0, "W_J1_1")], [])["J1"]
    assert c.update(5.0, [], [])["J1"] == []


def test_incident_warns_neighbours():
    c = Coordinator(PARAMS, ORDER)
    decision = SimpleNamespace(agent="incident", tls_id="J2", reason="lane blocked")
    events = c.update(10.0, [], [decision])
    assert [e.kind for e in events["J1"]] == ["incident"] and [e.kind for e in events["J3"]] == ["incident"]
    assert events["J2"] == [] and events["J1"][0].data["from_tls"] == "J2"


# --- spillback warning (core idea 5) ---
from dataclasses import replace  # noqa: E402

from flowmind.contracts import LaneState, make_example_state  # noqa: E402

SPILL = dict(PARAMS, spillback_occupancy=0.4)


def lane(lane_id, occupancy):
    return LaneState(lane_id, queue=0, vehicles=0, waiting_time=0.0, occupancy=occupancy)


def corridor_states(j1_j2=0.1, j2_j1=0.1):
    """J1 and J2 joined by two links: J1_J2_1 (eastbound) and J2_J1_1 (westbound)."""
    base = make_example_state()
    return {
        "J1": replace(base, tls_id="J1", lanes=(lane("J2_J1_1", j2_j1),), out_lanes=(lane("J1_J2_1", j1_j2),)),
        "J2": replace(base, tls_id="J2", lanes=(lane("J1_J2_1", j1_j2),), out_lanes=(lane("J2_J1_1", j2_j1),)),
        "J3": replace(base, tls_id="J3", lanes=(), out_lanes=()),
    }


def test_full_link_warns_the_junction_that_feeds_it():
    c = Coordinator(SPILL, ORDER)
    events = c.update(10.0, [], [], corridor_states(j1_j2=0.55))
    (e,) = events["J1"]
    assert e.kind == "incident" and e.data["cause"] == "spillback"
    assert e.data["from_tls"] == "J2" and e.data["occupancy"] == 0.55
    assert events["J2"] == [] and events["J3"] == []


def test_westbound_link_warns_the_other_junction():
    events = Coordinator(SPILL, ORDER).update(10.0, [], [], corridor_states(j2_j1=0.6))
    assert [e.data["cause"] for e in events["J2"]] == ["spillback"] and events["J1"] == []


def test_no_warning_below_threshold_or_without_states():
    c = Coordinator(SPILL, ORDER)
    assert all(v == [] for v in c.update(10.0, [], [], corridor_states(0.39, 0.39)).values())
    assert all(v == [] for v in c.update(10.0, [], []).values())
