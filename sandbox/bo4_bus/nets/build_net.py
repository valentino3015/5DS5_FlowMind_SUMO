"""
Build the BO4 bus scenario network (one signalized junction) reproducibly.

    python bo4_bus/nets/build_net.py

Writes (all in bo4_bus/nets/):
    bus.nod.xml, bus.edg.xml   plain XML inputs
    bus.tll.xml                static 6-phase signal program
    bus.net.xml                final network (netconvert output, committed)
    bus_stops.add.xml          4 bus stops (2 per arterial direction)

Layout (x to the east, y to the north, junction "C" at the origin):
    arterial W-E : 600 m each side, 2 lanes/direction, 13.89 m/s
    cross N-S    : 300 m each side, 1 lane/direction,  13.89 m/s
Signal (team defaults, cycle 75 s): arterial green 40 / yellow 3 / all-red 2 /
cross green 25 / yellow 3 / all-red 2. Green phases carry minDur=10 / maxDur=60
(ignored by the static program, used by adaptive controllers later).
Left turns are permitted (unprotected, 'g' = yield to opposing traffic).

Two netconvert passes: the first builds the topology, then the signal states
are derived from the connection directions (sumolib) and written as a tlLogic
that the second pass embeds into the net.
"""
import os
import subprocess
import sys

if "SUMO_HOME" not in os.environ:
    for _p in (r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Program Files\Eclipse\Sumo"):
        if os.path.isdir(_p):
            os.environ["SUMO_HOME"] = _p
            break

import sumolib

HERE = os.path.dirname(os.path.abspath(__file__))

ART_HALF = 600.0      # m, arterial length each side of the junction
CROSS_HALF = 300.0    # m, cross street length each side
SPEED = 13.89         # m/s (50 km/h), both streets
TLS_ID = "C"

ART_GREEN, CROSS_GREEN = 40, 25
YELLOW, ALL_RED = 3, 2          # s, team defaults
MIN_GREEN, MAX_GREEN = 10, 60   # s, team defaults

STOP_DIST = 150.0     # m, distance of the stop centre from the junction
STOP_LEN = 30.0       # m

ARTERIAL_IN = ("W2C", "E2C")
CROSS_IN = ("N2C", "S2C")

# bus stops: id -> (edge, "upstream"/"downstream")
STOPS = {
    "EB_up": ("W2C", "upstream"),
    "EB_down": ("C2E", "downstream"),
    "WB_up": ("E2C", "upstream"),
    "WB_down": ("C2W", "downstream"),
}


def p(name):
    return os.path.join(HERE, name)


def write_plain_xml():
    nodes = [
        ("C", 0, 0, "traffic_light"),
        ("W", -ART_HALF, 0, "priority"),
        ("E", ART_HALF, 0, "priority"),
        ("N", 0, CROSS_HALF, "priority"),
        ("S", 0, -CROSS_HALF, "priority"),
    ]
    with open(p("bus.nod.xml"), "w") as f:
        f.write("<nodes>\n")
        for nid, x, y, t in nodes:
            extra = ' tlType="static"' if t == "traffic_light" else ""
            f.write(f'    <node id="{nid}" x="{x:.2f}" y="{y:.2f}" type="{t}"{extra}/>\n')
        f.write("</nodes>\n")

    edges = [
        ("W2C", "W", "C", 2), ("C2E", "C", "E", 2),
        ("E2C", "E", "C", 2), ("C2W", "C", "W", 2),
        ("N2C", "N", "C", 1), ("C2N", "C", "N", 1),
        ("S2C", "S", "C", 1), ("C2S", "C", "S", 1),
    ]
    with open(p("bus.edg.xml"), "w") as f:
        f.write("<edges>\n")
        for eid, fr, to, lanes in edges:
            prio = 3 if lanes == 2 else 2
            f.write(f'    <edge id="{eid}" from="{fr}" to="{to}" numLanes="{lanes}" '
                    f'speed="{SPEED}" priority="{prio}"/>\n')
        f.write("</edges>\n")


def netconvert(*args):
    cmd = [sumolib.checkBinary("netconvert"), *args]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def write_tll(net_path):
    """Derive the 4 phase states from the connections controlled by the TLS."""
    net = sumolib.net.readNet(net_path, withInternal=False)
    tls = net.getTLS(TLS_ID)
    n_links = 1 + max(idx for _, _, idx in tls.getConnections())
    arterial = [None] * n_links   # True -> arterial movement
    is_left = [False] * n_links
    for in_lane, out_lane, idx in tls.getConnections():
        edge_id = in_lane.getEdge().getID()
        arterial[idx] = edge_id in ARTERIAL_IN
        conn = [c for c in in_lane.getOutgoing() if c.getToLane() == out_lane][0]
        is_left[idx] = conn.getDirection() in ("l", "L", "t")

    def green(for_arterial):
        return "".join(
            ("g" if is_left[i] else "G") if arterial[i] == for_arterial else "r"
            for i in range(n_links))

    def yellow(for_arterial):
        return "".join("y" if arterial[i] == for_arterial else "r" for i in range(n_links))

    all_red = "r" * n_links
    phases = [
        (ART_GREEN, green(True)), (YELLOW, yellow(True)), (ALL_RED, all_red),
        (CROSS_GREEN, green(False)), (YELLOW, yellow(False)), (ALL_RED, all_red),
    ]
    assert all(MIN_GREEN <= d <= MAX_GREEN for d in (ART_GREEN, CROSS_GREEN))
    with open(p("bus.tll.xml"), "w") as f:
        f.write("<additional>\n")
        f.write(f'    <tlLogic id="{TLS_ID}" type="static" programID="0" offset="0">\n')
        for dur, state in phases:
            bounds = (f' minDur="{MIN_GREEN}" maxDur="{MAX_GREEN}"'
                      if "G" in state or "g" in state else "")
            f.write(f'        <phase duration="{dur}" state="{state}"{bounds}/>\n')
        f.write("    </tlLogic>\n</additional>\n")
    return phases


def write_bus_stops(net_path):
    net = sumolib.net.readNet(net_path)
    with open(p("bus_stops.add.xml"), "w") as f:
        f.write("<additional>\n")
        for sid, (edge, where) in STOPS.items():
            lane = net.getEdge(edge).getLane(0)   # rightmost lane
            length = lane.getLength()
            centre = length - STOP_DIST if where == "upstream" else STOP_DIST
            start, end = centre - STOP_LEN / 2, centre + STOP_LEN / 2
            f.write(f'    <busStop id="{sid}" lane="{lane.getID()}" startPos="{start:.2f}" '
                    f'endPos="{end:.2f}" lines="L1"/>\n')
        f.write("</additional>\n")


def main():
    write_plain_xml()
    tmp = p("_tmp.net.xml")
    common = ["--no-turnarounds", "true", "--tls.default-type", "static",
              "--junctions.join", "false", "--no-warnings", "true"]
    netconvert("-n", p("bus.nod.xml"), "-e", p("bus.edg.xml"), "-o", tmp, *common)
    phases = write_tll(tmp)
    netconvert("--sumo-net-file", tmp, "--tllogic-files", p("bus.tll.xml"),
               "-o", p("bus.net.xml"), *common)
    os.remove(tmp)
    write_bus_stops(p("bus.net.xml"))
    for dur, state in phases:
        print(f"  phase {dur:>3}s  {state}")
    print("cycle:", sum(d for d, _ in phases), "s")


if __name__ == "__main__":
    sys.exit(main())
