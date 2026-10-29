"""Tests for the team part of the dashboard: per-agent live figures and final results (no SUMO needed).

Owner: Adem.
"""
from dataclasses import replace
from types import SimpleNamespace

from flowmind.contracts import Crossing, Event, SpecialVehicle, make_example_state
from flowmind.evaluation import TEAM_AGENTS, team_kpis, team_live


def test_team_live_counts_agents_vehicles_people_and_warnings():
    st = replace(make_example_state(),
                 special_vehicles=(SpecialVehicle("amb_1", "emergency", "north_in_0", 50.0, 10.0, None),
                                   SpecialVehicle("bus_we.1", "bus", "east_in_0", 80.0, 8.0, None)),
                 crossings=(Crossing(":J1_c0", 3, 25.0, 0), Crossing(":J1_c1", 1, 40.0, 0)))
    events = {"J1": [Event("incident", "J1", 99.0, {"cause": "spillback", "from_tls": "J2"}),
                     Event("incident", "J1", 99.0, {"cause": "incident", "from_tls": "J2"})]}
    decisions = [SimpleNamespace(agent="emergency"), SimpleNamespace(agent="flow"), SimpleNamespace(agent="flow")]
    live = team_live({"J1": st}, events, decisions, shield_changes=2)
    assert {f"ctrl_{a}" for a in TEAM_AGENTS} <= live.keys()
    assert live["ctrl_emergency"] == 1 and live["ctrl_flow"] == 2 and live["ctrl_bus"] == 0
    assert live["emergency_near"] == 1 and live["buses_near"] == 1
    assert live["ped_waiting"] == 4 and live["ped_wait_now_max_s"] == 40.0
    assert live["spillback_warnings"] == 1 and live["incident_warnings"] == 1
    assert live["shield_changes"] == 2


def test_team_live_has_the_same_columns_before_the_first_decision():
    # the dashboard's CSV header is written from the first sample, so the keys must never change
    assert team_live({}, {}, [], 0).keys() == team_live({"J1": make_example_state()}, {}, [], 0).keys()


def test_team_kpis_ignore_unfinished_trips(tmp_path):
    xml = tmp_path / "tripinfo.xml"
    xml.write_text("""<tripinfos>
  <tripinfo id="car.1" vType="car" arrival="100" duration="60" waitingTime="10"/>
  <tripinfo id="bus_we.1" vType="bus" arrival="200" duration="120" waitingTime="30"/>
  <tripinfo id="amb_1" vType="ambulance" arrival="300" duration="62" waitingTime="0"/>
  <tripinfo id="car.2" vType="car" arrival="-1" duration="500" waitingTime="400"/>
  <personinfo id="ped.1" duration="200" waitingTime="8"/>
  <personinfo id="ped.2" duration="150" waitingTime="20"/>
</tripinfos>""", encoding="utf-8")
    k = team_kpis(xml)
    assert k["team_arrived"] == 3                      # car.2 has not arrived
    assert k["team_mean_waiting_s"] == round(40 / 3, 2)
    assert k["bus_trips"] == 1 and k["bus_mean_trip_s"] == 120
    assert k["ambulance_trip_s"] == {"amb_1": 62.0}
    assert k["pedestrians"] == 2 and k["ped_mean_wait_s"] == 14 and k["ped_max_wait_s"] == 20
