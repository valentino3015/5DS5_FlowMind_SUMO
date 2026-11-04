"""Smoke test: the 10-junction grid (scenarios/grid10/, 2 streets x 5 junctions) runs with the env
and the coordinator finds its neighbours from the roads.

Owner: Adem. Needs SUMO (installed by requirements.txt). Takes a few seconds.
"""
from pathlib import Path

import yaml

from flowmind.control.coordinator import Coordinator, road_links
from flowmind.env import SumoCorridorEnv

ROOT = Path(__file__).parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
NET = ROOT / "scenarios" / "grid10" / "grid10.net.xml"
DEMO = ROOT / "scenarios" / "grid10" / "grid10_demo.rou.xml"
JUNCTIONS = [f"J{i}" for i in range(1, 11)]


def make_env():
    return SumoCorridorEnv(config=CONFIG, net_file=NET, route_file=DEMO, seed=1, begin_time=0)


def test_ten_junctions_stops_and_links():
    import traci
    env = make_env()
    try:
        states = env.start()
        assert sorted(states, key=lambda t: int(t[1:])) == JUNCTIONS
        assert len(traci.busstop.getIDList()) == 16        # 4 per direction on each street
        links = set(road_links(states))
        assert {("J1", "J2"), ("J2", "J1"), ("J1", "J6"), ("J6", "J1"), ("J6", "J7")} <= links
        assert ("J1", "J7") not in links                     # no diagonal road
        assert len(links) == 2 * (4 + 4 + 5)                 # 2 streets x 4 links + 5 cross streets, both ways
    finally:
        env.close()


def test_ambulance_green_corridor_turns_the_corner():
    # amb_1 (60 s): top street J1-J2-J3, down at J3 to J8, bottom street J9-J10.
    env = make_env()
    warned = set()
    try:
        states = env.start()
        listed = [t for t in CONFIG["corridor"]["order"] if t in states]
        coordinator = Coordinator(CONFIG["coordinator"], listed + [t for t in states if t not in listed], road_links(states))
        for _ in range(40):  # up to 200 s
            states, _, _ = env.step({})
            now = next(iter(states.values())).time
            events = coordinator.update(now, env.emergency_approaches(), [], states)
            warned |= {tls for tls, evs in events.items() if any(e.kind == "emergency" for e in evs)}
            if {"J8", "J9", "J10"} <= warned:
                break
        assert {"J8", "J9", "J10"} <= warned, f"only warned {sorted(warned)}"
        assert not warned & {"J4", "J5", "J6", "J7"}         # not on its route
    finally:
        env.close()
