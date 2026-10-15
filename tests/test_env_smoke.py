"""Smoke test: Ahmed's env runs on our corridor (buses, ambulance, pedestrians).

Owner: Adem. Needs SUMO (installed by requirements.txt). Takes a few seconds.
"""
from pathlib import Path

import yaml

from flowmind.contracts import JunctionState
from flowmind.env import SumoCorridorEnv

ROOT = Path(__file__).parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
NET = ROOT / "scenarios" / "corridor" / "corridor.net.xml"
ROUTES = ROOT / "scenarios" / "corridor" / "corridor.rou.xml"


def make_env(seed):
    return SumoCorridorEnv(config=CONFIG, net_file=NET, route_file=ROUTES, seed=seed, begin_time=0)


def test_states_have_valid_phases_and_crossings():
    env = make_env(seed=1)
    try:
        states = env.start()
        assert sorted(states) == ["J1", "J2", "J3"]
        for _ in range(12):  # 60 s with the default phase
            states, done, info = env.step({})
        for s in states.values():
            assert isinstance(s, JunctionState)
            assert s.num_phases == len(s.phase_lanes) == len(s.phase_crossings) >= 2
            lane_ids = {lane.lane_id for lane in s.lanes}
            for lanes in s.phase_lanes:
                assert set(lanes) <= lane_ids            # only vehicle lanes, no walking areas
            assert len(s.crossings) == 4                 # 4 pedestrian crossings per junction
            crossing_ids = {c.crossing_id for c in s.crossings}
            for ids in s.phase_crossings:
                assert set(ids) <= crossing_ids
    finally:
        env.close()


def test_ambulance_is_announced_to_the_coordinator():
    env = make_env(seed=1)
    try:
        env.start()
        approaches = []
        for _ in range(40):  # up to 200 s; the ambulance departs at 120 s
            env.step({})
            approaches = env.emergency_approaches()
            if approaches:
                break
        assert approaches, "amb_1 (departs at 120 s) should be announced"
        vehicle_id, tls_id, distance, speed, lane = approaches[0]
        assert vehicle_id == "amb_1" and tls_id in ("J1", "J2", "J3")
        assert distance > 0 and lane.startswith("W_J1")
    finally:
        env.close()


def test_seed_changes_traffic():
    vehicles = []
    for seed in (1, 2):
        env = make_env(seed)
        try:
            env.start()
            for _ in range(40):  # 200 s
                _, _, info = env.step({})
            vehicles.append((info["vehicle_count"], info["halting_vehicles"]))
        finally:
            env.close()
    assert vehicles[0] != vehicles[1]
