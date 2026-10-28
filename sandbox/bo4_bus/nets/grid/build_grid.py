"""
Build the BO4 3x3 grid scenario reproducibly.

    python bo4_bus/nets/grid/build_grid.py

Writes (all in bo4_bus/nets/grid/):
    grid.tll.xml          fixed-time programs of the 9 signals
    grid.net.xml          final network (committed)
    grid_stops.add.xml    bus stops of lines L1 (E-W, middle row) and L2 (N-S, middle column)

Network (netgenerate): 3x3 junctions A0..C2 (letter = column, digit = row), 300 m
blocks, 300 m attached fringe streets, 2 lanes/direction, 13.89 m/s and one ~50 m
left-turn lane on every approach. netgenerate implements the turn lane by splitting
each approach at 250 m: edge "XY" (2 lanes) + edge "XY.250.00" (3 lanes, lane 2 is
the left-turn lane). Traffic lights at the 9 inner junctions only.

Signals (team defaults, cycle 75 s, offset 0 everywhere):
    E-W green 35 / yellow 3 / all-red 2 / N-S green 30 / yellow 3 / all-red 2
Left turns are permitted, unprotected ('g'); greens carry minDur 10 / maxDur 60.

Bus stops: 30 m, rightmost lane. Per line direction: one on the entry link (centre
150 m after the fringe node) and one ~100 m after each of the 3 junctions crossed
(far-side, 85-115 m on the downstream edge) -> 4 stops per direction.
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
NET = os.path.join(HERE, "grid.net.xml")
STOPS_ADD = os.path.join(HERE, "grid_stops.add.xml")
TLL = os.path.join(HERE, "grid.tll.xml")

BLOCK = 300.0
SPEED = 13.89
TURN_LANE_LEN = 50.0
TLS_IDS = [f"{c}{r}" for c in "ABC" for r in "012"]

EW_GREEN, NS_GREEN = 35, 30
YELLOW, ALL_RED = 3, 2          # s, team defaults
MIN_GREEN, MAX_GREEN = 10, 60   # s, team defaults

STOP_LEN = 30.0
ENTRY_STOP_CENTRE = 150.0       # m from the start of the entry link
FAR_SIDE_STOP_CENTRE = 100.0    # m after the junction


def _approach(a, b):
    """Edges from node a to node b: the 2-lane piece and the 3-lane turn-lane piece."""
    return [f"{a}{b}", f"{a}{b}.250.00"]


# line direction -> (node sequence fringe..fringe)
LINE_NODES = {
    "L1_EB": ["left1", "A1", "B1", "C1", "right1"],
    "L1_WB": ["right1", "C1", "B1", "A1", "left1"],
    "L2_NB": ["bottom1", "B0", "B1", "B2", "top1"],
    "L2_SB": ["top1", "B2", "B1", "B0", "bottom1"],
}


def line_edges(nodes):
    """Edge list of a line: split approach pieces into every junction, plain exit edge."""
    edges = []
    for a, b in zip(nodes, nodes[1:]):
        edges += [f"{a}{b}"] if b == nodes[-1] else _approach(a, b)
    return edges


def line_stops(direction):
    """Ordered stop ids of a line direction: entry stop, then one after each junction."""
    junctions = LINE_NODES[direction][1:-1]
    return [f"{direction}_s0"] + [f"{direction}_s{i + 1}" for i in range(len(junctions))]


LINES = {d: {"edges": line_edges(n), "stops": line_stops(d), "junctions": n[1:-1]}
         for d, n in LINE_NODES.items()}


def _stop_edges(direction):
    """Edge carrying each stop of `line_stops(direction)` (all first pieces of a link)."""
    n = LINE_NODES[direction]
    return [f"{a}{b}" for a, b in zip(n, n[1:])]


def run(*cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def write_tll(net_path):
    net = sumolib.net.readNet(net_path)
    with open(TLL, "w") as f:
        f.write("<additional>\n")
        for tid in TLS_IDS:
            tls = net.getTLS(tid)
            n_links = 1 + max(idx for _, _, idx in tls.getConnections())
            ew = [None] * n_links
            left = [False] * n_links
            for in_lane, out_lane, idx in tls.getConnections():
                edge = in_lane.getEdge()
                (x0, y0), (x1, y1) = edge.getFromNode().getCoord(), edge.getToNode().getCoord()
                ew[idx] = abs(x1 - x0) > abs(y1 - y0)
                conn = [c for c in in_lane.getOutgoing() if c.getToLane() == out_lane][0]
                left[idx] = conn.getDirection() in ("l", "L", "t")

            def green(is_ew):
                return "".join(("g" if left[i] else "G") if ew[i] == is_ew else "r"
                               for i in range(n_links))

            def yellow(is_ew):
                return "".join("y" if ew[i] == is_ew else "r" for i in range(n_links))

            red = "r" * n_links
            phases = [(EW_GREEN, green(True)), (YELLOW, yellow(True)), (ALL_RED, red),
                      (NS_GREEN, green(False)), (YELLOW, yellow(False)), (ALL_RED, red)]
            f.write(f'    <tlLogic id="{tid}" type="static" programID="0" offset="0">\n')
            for dur, state in phases:
                bounds = (f' minDur="{MIN_GREEN}" maxDur="{MAX_GREEN}"'
                          if "G" in state or "g" in state else "")
                f.write(f'        <phase duration="{dur}" state="{state}"{bounds}/>\n')
            f.write("    </tlLogic>\n")
        f.write("</additional>\n")
    assert all(MIN_GREEN <= g <= MAX_GREEN for g in (EW_GREEN, NS_GREEN))
    return EW_GREEN + NS_GREEN + 2 * (YELLOW + ALL_RED)


def write_stops(net_path):
    net = sumolib.net.readNet(net_path)
    with open(STOPS_ADD, "w") as f:
        f.write("<additional>\n")
        for d in LINES:
            line = d.split("_")[0]
            for k, (sid, edge) in enumerate(zip(line_stops(d), _stop_edges(d))):
                lane = net.getEdge(edge).getLane(0)
                centre = ENTRY_STOP_CENTRE if k == 0 else FAR_SIDE_STOP_CENTRE
                start, end = centre - STOP_LEN / 2, centre + STOP_LEN / 2
                assert end < lane.getLength(), (sid, lane.getLength())
                f.write(f'    <busStop id="{sid}" lane="{lane.getID()}" startPos="{start:.2f}" '
                        f'endPos="{end:.2f}" lines="{line}"/>\n')
        f.write("</additional>\n")


def main():
    tmp = os.path.join(HERE, "_tmp.net.xml")
    run(sumolib.checkBinary("netgenerate"), "--grid", "--grid.number", "3",
        "--grid.length", str(BLOCK), "--grid.attach-length", str(BLOCK),
        "--default.lanenumber", "2", "--default.speed", str(SPEED),
        "--turn-lanes", "1", "--turn-lanes.length", str(TURN_LANE_LEN),
        "--tls.set", ",".join(TLS_IDS), "--tls.default-type", "static",
        "--no-turnarounds", "true", "--no-warnings", "true", "-o", tmp)
    cycle = write_tll(tmp)
    run(sumolib.checkBinary("netconvert"), "--sumo-net-file", tmp, "--tllogic-files", TLL,
        "--no-warnings", "true", "-o", NET)
    os.remove(tmp)
    write_stops(NET)
    for d, spec in LINES.items():
        print(f"{d}: stops {spec['stops']}\n       edges {' '.join(spec['edges'])}")
    print("cycle:", cycle, "s")


if __name__ == "__main__":
    sys.exit(main())
