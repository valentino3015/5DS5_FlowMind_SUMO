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
APPROACHES = ("A1B1", "C1B1", "B0B1", "B2B1")   # fixed order: from W, E, S, N
NEIGHBOURS = ("A1", "C1", "B0", "B2")
EW_GREEN_IDX, NS_GREEN_IDX = 0, 3               # program phase indices of the 2 greens
GREEN_PROGRAM_IDX = (EW_GREEN_IDX, NS_GREEN_IDX)  # action k -> program phase
BUS_PREFIXES = tuple(f"{d}_" for d in LINES)


def line_of(bus_id):
    return bus_id.rsplit("_", 1)[0]


def approach_of(lane_id):
    """'A1B1.250.00_2' / 'A1B1_0' -> 'A1B1'."""
    return lane_id.rsplit("_", 1)[0].split(".")[0]


class Geometry:
    """Incoming / outgoing lanes of one junction (sumolib, once)."""

    def __init__(self, net, tls=TLS, approaches=APPROACHES):
        self.tls = tls
        self.approaches = approaches
        self.lanes_in = []           # (lane id, length, offset: distance from lane end to stop line)
        self.edge_dist = {}          # edge id -> (length, offset)
        for a in approaches:
            main, piece = net.getEdge(a), net.getEdge(a + ".250.00")
            lp = piece.getLength()
            for edge, off in ((piece, 0.0), (main, lp)):
                self.edge_dist[edge.getID()] = (edge.getLength(), off)
                for lane in edge.getLanes():
                    self.lanes_in.append((lane.getID(), lane.getLength(), off))
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
        for v in self._prev_buses - now:            # left the approach: passed the junction
            self._last_pass[line_of(v)] = t
        self._prev_buses = now

        approaching = []
        for v, lid, dist in sorted(buses, key=lambda b: b[2]):
            last = self._last_pass.get(line_of(v))
            approaching.append(VehicleInfo(
                vehicle_id=v, vclass="bus", lane_id=lid, distance_to_stopline=dist,
                speed=sim.vehicle.getSpeed(v), waiting_time=sim.vehicle.getWaitingTime(v),
                schedule_delay=None, persons_on_board=self._occ[v],
                line_id=line_of(v),
                headway_lateness=None if last is None else t - last - self.headway))

        return JunctionState(tls_id=tls, sim_time=t, green_phases=self.green_states,
                             current_green=current_green, time_in_phase=time_in_phase,
                             in_transition=in_transition, incoming=incoming, outgoing=outgoing,
                             approaching=tuple(approaching), crossings=(), neighbours=NEIGHBOURS)
