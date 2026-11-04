"""Smoke test: the 5-junction corridor (scenarios/corridor5/) runs with the env and the coordinator.

Owner: Adem. Needs SUMO (installed by requirements.txt). Takes a few seconds.
"""
from pathlib import Path

import yaml

from flowmind.control.coordinator import Coordinator
from flowmind.env import SumoCorridorEnv

ROOT = Path(__file__).parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
NET = ROOT / "scenarios" / "corridor5" / "corridor5.net.xml"
DEMO = ROOT / "scenarios" / "corridor5" / "corridor5_demo.rou.xml"


def test_five_junctions_with_crossings():
    env = SumoCorridorEnv(config=CONFIG, net_file=NET, route_file=DEMO, seed=1, begin_time=0)
    try:
        states = env.start()
        assert sorted(states) == ["J1", "J2", "J3", "J4", "J5"]
        for s in states.values():
            assert s.num_phases == len(s.phase_lanes) == len(s.phase_crossings) >= 2
            assert len(s.crossings) == 4
    finally:
        env.close()


def test_coordinator_warns_the_far_junctions_about_the_ambulance():
    # amb_1 departs at 60 s from the west; the coordinator follows corridor.order (J1..J5).
    env = SumoCorridorEnv(config=CONFIG, net_file=NET, route_file=DEMO, seed=1, begin_time=0)
    order = [t for t in CONFIG["corridor"]["order"]]
    coordinator = Coordinator(CONFIG["coordinator"], order)
    warned = set()
    try:
        states = env.start()
        assert order == ["J1", "J2", "J3", "J4", "J5"]
        for _ in range(40):  # up to 200 s
            states, _, _ = env.step({})
            now = next(iter(states.values())).time
            events = coordinator.update(now, env.emergency_approaches(), [], states)
            warned |= {tls for tls, evs in events.items() if any(e.kind == "emergency" for e in evs)}
            if {"J4", "J5"} <= warned:
                break
        assert {"J4", "J5"} <= warned, f"only warned {sorted(warned)}"
    finally:
        env.close()


def test_bus_stop_zones_are_loaded():
    import traci
    env = SumoCorridorEnv(config=CONFIG, net_file=NET, route_file=DEMO, seed=1, begin_time=0)
    try:
        env.start()
        assert len(traci.busstop.getIDList()) == 8      # 4 per direction, after each junction
    finally:
        env.close()
