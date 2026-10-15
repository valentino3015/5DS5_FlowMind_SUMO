"""Smoke test: SUMO starts on the corridor and env.py returns valid states.

Owner: Adem. Needs SUMO (installed by requirements.txt). Takes a few seconds.
"""
from pathlib import Path

import yaml

from flowmind.contracts import JunctionState
from flowmind.env import CorridorEnv

CONFIG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())


def test_corridor_runs_and_states_are_valid():
    env = CorridorEnv(CONFIG, seed=1)
    try:
        states = env.reset()
        assert sorted(states) == ["J1", "J2", "J3"]
        for _ in range(12):  # 60 s, switching phase every 30 s
            states = env.step({t: int(env.time() // 30) % s.num_phases for t, s in states.items()})
        for s in states.values():
            assert isinstance(s, JunctionState)
            assert s.num_phases == len(s.phase_lanes) == len(s.phase_crossings) >= 2
            assert 0 <= s.phase < s.num_phases
            lane_ids = {lane.lane_id for lane in s.lanes}
            for lanes in s.phase_lanes:
                assert set(lanes) <= lane_ids
        assert env.time() == 60
    finally:
        env.close()


def test_seed_changes_traffic():
    arrived = []
    for seed in (1, 2):
        env = CorridorEnv(CONFIG, seed=seed)
        try:
            env.reset()
            for _ in range(40):  # 200 s
                env.step()
            arrived.append(env.summary()["total_waiting_veh_s"])
        finally:
            env.close()
    assert arrived[0] != arrived[1]
