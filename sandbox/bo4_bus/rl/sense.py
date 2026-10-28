"""
Sensor for the RL junction: builds one JunctionState (contract.py) from libsumo per
decision tick. Besides env.py this is the ONLY file in bo4_bus/rl/ that calls SUMO.

Static geometry comes from the .net.xml (sumolib) once; the per-tick reads are one
batch of lane / vehicle getters. The sensor keeps one piece of history: the time a bus
of each line last passed the junction, for the headway lateness of the next bus
(same definition as controller C in ../rule_tsp.py, at tick resolution).
"""
import os
import sys

from contract import JunctionState, LaneState, VehicleInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "nets", "grid"))
from build_grid import LINES  # noqa: E402

TLS = "B1"
APPROACHES = ("A1B1", "C1B1", "B0B1", "B2B1")   # B1's approaches: from W, E, S, N
NEIGHBOURS = ("A1", "C1", "B0", "B2")
COMPASS = ("W", "E", "S", "N")                  # order of the approach slots, every junction
EW_GREEN_IDX, NS_GREEN_IDX = 0, 3               # program phase indices of the 2 greens
GREEN_PROGRAM_IDX = (EW_GREEN_IDX, NS_GREEN_IDX)  # action k -> program phase
BUS_PREFIXES = tuple(f"{d}_" for d in LINES)
PIECE = ".250.00"                               # netgenerate's turn-lane piece of an approach


def line_of(bus_id):
    return bus_id.rsplit("_", 1)[0]


def approach_of(lane_id):
    """'A1B1.250.00_2' / 'A1B1_0' -> 'A1B1'."""
    return lane_id.rsplit("_", 1)[0].split(".")[0]


def compass_of(net, from_node, tls):
    """Side of junction `tls` that an approach from `from_node` comes from: W/E/S/N."""
    (x0, y0), (x1, y1) = net.getNode(from_node).getCoord(), net.getNode(tls).getCoord()
    dx, dy = x0 - x1, y0 - y1
    if abs(dx) >= abs(dy):
        return "W" if dx < 0 else "E"
    return "S" if dy < 0 else "N"


def junction_approaches(net, tls):
    """{W/E/S/N: approach id} of junction `tls` from the network (sides without an
    approach are absent). Approach id = the upstream (main) edge, e.g. 'left1A1'."""
    out = {}
    for e in net.getNode(tls).getIncoming():
        if e.getID().startswith(":"):
            continue
        main = e.getID()[:-len(PIECE)] if e.getID().endswith(PIECE) else e.getID()
        main_edge = net.getEdge(main)
        side = compass_of(net, main_edge.getFromNode().getID(), tls)
        assert side not in out, f"{tls}: two approaches from {side}"
        out[side] = main
    return out


class Geometry:
    """Incoming / outgoing lanes of one junction (sumolib, once), for any junction.

    approaches: one approach id (or None if the side has no approach) per COMPASS slot,
    W, E, S, N. Default: read from the network (junction_approaches); for B1 this is
    exactly APPROACHES. An approach is its main edge plus, if present, the turn-lane
    piece main + '.250.00' that ends at the stop line.
    straight: approach id -> outgoing edge reached by its straight ('s') connections.
    neighbours: controlled junctions at the upstream end of the approaches.
    """

    def __init__(self, net, tls=TLS, approaches=None):
        self.tls = tls
        if approaches is None:
            sides = junction_approaches(net, tls)
            approaches = tuple(sides.get(s) for s in COMPASS)
        self.approaches = tuple(approaches)
        self.lanes_in = []           # (lane id, length, offset: distance from lane end to stop line)
        self.edge_dist = {}          # edge id -> (length, offset)
        self.straight = {}
        neighbours = []
        for a in self.approaches:
            if a is None:
                continue
            main = net.getEdge(a)
            piece = net.getEdge(a + PIECE) if net.hasEdge(a + PIECE) else None
            chain = ((piece, 0.0), (main, piece.getLength())) if piece else ((main, 0.0),)
            for edge, off in chain:
                self.edge_dist[edge.getID()] = (edge.getLength(), off)
                for lane in edge.getLanes():
                    self.lanes_in.append((lane.getID(), lane.getLength(), off))
            last = chain[0][0]                       # the edge that ends at the stop line
            straight = {to.getID() for to, conns in last.getOutgoing().items()
                        if any(c.getDirection() == "s" for c in conns)}
            assert len(straight) <= 1, f"{a}: several straight exits {straight}"
            if straight:
                self.straight[a] = straight.pop()
            frm = main.getFromNode()
            if frm.getType() == "traffic_light":
                neighbours.append(frm.getID())
        self.neighbours = tuple(neighbours)
        node = net.getNode(tls)
        self.lanes_out = [(lane.getID(), lane.getLength()) for e in node.getOutgoing()
                          if not e.getID().startswith(":") for lane in e.getLanes()]


class Sensor:
    """sense(sim, t) -> JunctionState for one junction."""

    def __init__(self, geometry, headway=180.0, detect_range=None):
        self.g = geometry
        self.headway = headway
        # whole approach by default (268 m), so every bus that contributes waiting time
        # on the incoming lanes is also in `approaching` (needed by the person reward)
        self.detect_range = detect_range or max(l + o for _, l, o in geometry.lanes_in)
        self.green_states = None
        self.reset()

    def reset(self):
        self._last_pass = {}         # line -> sim time a bus of it last left the approaches
        self._prev_buses = set()
        self._occ = {}
        self.passages = ()           # buses that passed during the last sense() interval

    def _lane_state(self, sim, lane_id, length):
        return LaneState(lane_id=lane_id, length=length,
                         vehicle_count=sim.lane.getLastStepVehicleNumber(lane_id),
                         halting_count=sim.lane.getLastStepHaltingNumber(lane_id),
                         occupancy=sim.lane.getLastStepOccupancy(lane_id) / 100.0,
                         mean_speed=sim.lane.getLastStepMeanSpeed(lane_id),
                         waiting_time=sim.lane.getWaitingTime(lane_id),
                         co2_mg=sim.lane.getCO2Emission(lane_id))

    def sense(self, sim, t, current_green, time_in_phase, in_transition):
        """current_green / time_in_phase / in_transition come from the env's signal
        controller, which owns the transitions of this junction."""
        tls = self.g.tls
        if self.green_states is None:
            logic = sim.trafficlight.getAllProgramLogics(tls)[0]
            self.green_states = tuple(logic.phases[i].state for i in GREEN_PROGRAM_IDX)

        incoming = tuple(self._lane_state(sim, lid, length) for lid, length, _ in self.g.lanes_in)
        outgoing = tuple(self._lane_state(sim, lid, length) for lid, length in self.g.lanes_out)

        buses = []
        for lid, length, off in self.g.lanes_in:
            for v in sim.lane.getLastStepVehicleIDs(lid):
                if not v.startswith(BUS_PREFIXES):
                    continue
                dist = length - sim.vehicle.getLanePosition(v) + off
                if dist > self.detect_range:
                    continue
                if v not in self._occ:
                    self._occ[v] = int(float(sim.vehicle.getParameter(v, "occupancy") or 0))
                buses.append((v, lid, dist))

        now = {v for v, _, _ in buses}
        passed = []
        for v in sorted(self._prev_buses - now):    # left the approach: passed the junction
            prev = self._last_pass.get(line_of(v))
            passed.append({"vehicle_id": v, "line_id": line_of(v), "t": t,
                           "gap": None if prev is None else t - prev,
                           "persons_on_board": self._occ.get(v, 0)})
            self._last_pass[line_of(v)] = t
        # events for reward G2: gap = time since the previous bus of the same line passed
        # this junction (None for the first one seen); not part of the JunctionState
        self.passages = tuple(passed)
        self._prev_buses = now

        approaching = []
        for v, lid, dist in sorted(buses, key=lambda b: b[2]):
            last = self._last_pass.get(line_of(v))
            approaching.append(VehicleInfo(
                vehicle_id=v, vclass="bus", lane_id=lid, distance_to_stopline=dist,
                speed=sim.vehicle.getSpeed(v), waiting_time=sim.vehicle.getWaitingTime(v),
                schedule_delay=None, persons_on_board=self._occ[v],
                line_id=line_of(v),
                headway_lateness=None if last is None else t - last - self.headway,
                at_stop=bool(sim.vehicle.isAtBusStop(v))))

        return JunctionState(tls_id=tls, sim_time=t, green_phases=self.green_states,
                             current_green=current_green, time_in_phase=time_in_phase,
                             in_transition=in_transition, incoming=incoming, outgoing=outgoing,
                             approaching=tuple(approaching), crossings=(), neighbours=self.g.neighbours)
