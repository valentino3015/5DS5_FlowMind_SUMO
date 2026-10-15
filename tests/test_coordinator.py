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
