"""The corridor route files must stay in departure order (no SUMO needed).

SUMO reads a route file in time order and silently drops an entry that starts earlier than
the one read before it (no error with --no-warnings): a misplaced event would simply vanish.

Owner: Adem.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

SCENARIOS = Path(__file__).parent.parent / "scenarios"
CORRIDOR = SCENARIOS / "corridor"


def starts(route_file):
    root = ET.parse(route_file).getroot()
    return [(e.get("id"), float(e.get("depart") or e.get("begin")))
            for e in root if e.tag in ("vehicle", "flow", "personFlow", "trip", "person")]


@pytest.mark.parametrize("name", ["corridor/corridor.rou.xml", "corridor/corridor_demo.rou.xml",
                                  "corridor5/corridor5.rou.xml", "corridor5/corridor5_demo.rou.xml",
                                  "grid10/grid10.rou.xml", "grid10/grid10_demo.rou.xml"])
def test_route_file_is_in_departure_order(name):
    entries = starts(SCENARIOS / name)
    for (a, ta), (b, tb) in zip(entries, entries[1:]):
        assert ta <= tb, f"{b} starts at {tb} s, before {a} ({ta} s): SUMO would drop it"


def test_demo_has_every_event():
    ids = {i for i, _ in starts(CORRIDOR / "corridor_demo.rou.xml")}
    assert {"amb_1", "amb_2", "accident_1", "accident_2", "bus_ew.0", "bus_ew.1", "bus_ew.2", "ped_rush"} <= ids


def test_long_corridor_demo_has_every_event():
    ids = {i for i, _ in starts(SCENARIOS / "corridor5" / "corridor5_demo.rou.xml")}
    assert {"amb_1", "amb_2", "accident_1", "accident_2", "bus_ew.0", "bus_ew.1", "bus_ew.2", "ped_rush"} <= ids


def test_grid_demo_has_every_event():
    ids = {i for i, _ in starts(SCENARIOS / "grid10" / "grid10_demo.rou.xml")}
    assert {"amb_1", "amb_2", "accident_1", "accident_2", "bus_ew.0", "bus_ew.1", "bus_ew.2", "ped_rush"} <= ids
