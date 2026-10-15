"""SUMO connection: the ONLY file that imports traci / libsumo.

Owner: Adem.

What it does:
- starts SUMO on a scenario from scenarios/ (seed and GUI from the caller)
- every step (sim.delta_time seconds) builds one JunctionState per traffic light
- applies the green phase chosen for each junction, inserting yellow
  (signals.yellow) and all-red (signals.all_red) automatically
- counts a few totals for the end-of-run summary

Usage (this is also the hook for training the flow agent):
    env = CorridorEnv(config, seed=1)
    states = env.reset()                       # dict: tls_id -> JunctionState
    while not env.done():
        states = env.step({"J1": 0, "J2": 1})  # green phase index per junction
    env.close()
"""
from __future__ import annotations

import os
from pathlib import Path

import sumo

os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)

from flowmind.contracts import Crossing, JunctionState, LaneState, SpecialVehicle  # noqa: E402

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"
HALT_SPEED = 0.1  # m/s: below this a vehicle or pedestrian counts as stopped (SUMO's own definition)


def is_green(c: str) -> bool:
    return c in "Gg"


class Junction:
    """Fixed facts about one traffic light, read once from SUMO at the start."""

    def __init__(self, api, tls_id: str):
        self.tls_id = tls_id
        self.links = api.trafficlight.getControlledLinks(tls_id)  # per signal index: [(in, out, via)]
        states = [p.state for p in api.trafficlight.getAllProgramLogics(tls_id)[0].phases]

        # Green phases = phases without yellow. Drop a phase whose greens are a subset of
        # another one (SUMO adds "same green, pedestrians already red" clearance phases).
        candidates = [s for s in states if "y" not in s and any(is_green(c) for c in s)]
        greens = lambda s: {i for i, c in enumerate(s) if is_green(c)}  # noqa: E731
        self.green_states = [s for s in candidates if not any(greens(s) < greens(o) for o in candidates)]

        self.in_lanes = self._lanes(0)
        self.out_lanes = self._lanes(1)
        self.crossings = sorted({self._crossing_id(link) for link in self.links if link and self._is_crossing(link)})
        self.walking_areas = [e for e in api.edge.getIDList() if e.startswith(f":{tls_id}_w")]
        self.phase_lanes = tuple(self._green_items(s, crossing=False) for s in self.green_states)
        self.phase_crossings = tuple(self._green_items(s, crossing=True) for s in self.green_states)

    @staticmethod
    def _is_crossing(link) -> bool:
        return link[0][0].startswith(":")  # pedestrian link: walking area ":J1_w1_0" -> crossing ":J1_c0_0"

    @staticmethod
    def _crossing_id(link) -> str:
        return link[0][1].rsplit("_", 1)[0]  # crossing lane ":J1_c0_0" -> crossing id ":J1_c0"

    def _lanes(self, pos: int) -> tuple[str, ...]:
        lanes = [link[0][pos] for link in self.links if link and not self._is_crossing(link)]
        return tuple(dict.fromkeys(lanes))  # unique, keep order

    def _green_items(self, state: str, crossing: bool) -> tuple[str, ...]:
        items = []
        for i, link in enumerate(self.links):
            if link and is_green(state[i]) and self._is_crossing(link) == crossing:
                items.append(self._crossing_id(link) if crossing else link[0][0])
        return tuple(dict.fromkeys(items))


class CorridorEnv:
    """Runs SUMO and translates between SUMO and our contracts."""

    def __init__(self, config: dict, scenario: str = "corridor", seed: int = 0, gui: bool = False):
        self.cfg = config
        self.scenario_dir = SCENARIOS / scenario
        self.sumocfg = self.scenario_dir / f"{scenario}.sumocfg"
        self.seed = seed
        self.gui = gui
        if gui:
            import traci as api  # sumo-gui needs the TraCI socket connection
        else:
            import libsumo as api  # much faster, no GUI
        self.api = api
        self.running = False

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> dict[str, JunctionState]:
        """(Re)start SUMO and return the first states."""
        if self.running:
            self.close()
        binary = Path(sumo.SUMO_HOME) / "bin" / ("sumo-gui" if self.gui else "sumo")
        cmd = [str(binary), "-c", str(self.sumocfg), "--seed", str(self.seed)]
        if self.gui:
            cmd += ["--start", "--quit-on-end", "--delay", str(self.cfg["sim"]["gui_delay"]),
                    "--window-size", "1400,750"]
            settings = self.scenario_dir / "gui-settings.xml"
            if settings.exists():
                cmd += ["--gui-settings-file", str(settings)]
        self.api.start(cmd)
        self.running = True
        if self.gui:  # zoom the view on the whole street map
            (x1, y1), (x2, y2) = self.api.simulation.getNetBoundary()
            self.api.gui.setBoundary("View #0", x1, y1, x2, y2)

        self.junctions = {t: Junction(self.api, t) for t in self.api.trafficlight.getIDList()}
        self.phase = {t: 0 for t in self.junctions}          # current green phase index
        self.time_in_phase = {t: 0.0 for t in self.junctions}
        self.transition = {t: [] for t in self.junctions}   # queue of [state string, seconds left]
        self.target = {t: 0 for t in self.junctions}
        for t, j in self.junctions.items():
            self.api.trafficlight.setRedYellowGreenState(t, j.green_states[0])
        self.pois = set()  # overlay labels already created (GUI only)
        self.totals = {"waiting_veh_s": 0.0, "arrived": 0, "emergency_depart": {}, "emergency_travel": {}}
        return self.states()

    def close(self):
        if self.running:
            self.api.close()
            self.running = False

    def time(self) -> float:
        return self.api.simulation.getTime()

    def done(self) -> bool:
        return self.time() >= self.cfg["sim"]["seconds"] or self.api.simulation.getMinExpectedNumber() == 0

    # ------------------------------------------------------------------ control
    def apply(self, tls_id: str, green: int):
        """Ask for green phase `green`. Yellow + all-red are inserted when it changes."""
        if self.transition[tls_id] or green == self.phase[tls_id]:
            return  # already switching, or nothing to change
        old = self.junctions[tls_id].green_states[self.phase[tls_id]]
        new = self.junctions[tls_id].green_states[green]
        yellow = "".join(("G" if is_green(n) else "y") if is_green(o) else "r" for o, n in zip(old, new))
        all_red = "r" * len(old)
        self.transition[tls_id] = [[yellow, self.cfg["signals"]["yellow"]],
                                   [all_red, self.cfg["signals"]["all_red"]]]
        self.target[tls_id] = green
        self.api.trafficlight.setRedYellowGreenState(tls_id, yellow)

    def step(self, actions: dict[str, int] | None = None) -> dict[str, JunctionState]:
        """Apply the actions, run sim.delta_time seconds, return the new states."""
        for tls_id, green in (actions or {}).items():
            self.apply(tls_id, green)
        for _ in range(int(self.cfg["sim"]["delta_time"])):
            self.api.simulationStep()
            self._tick()
            self._count()
        return self.states()

    def _tick(self):
        """One second passed: advance the yellow / all-red sequences."""
        for t, queue in self.transition.items():
            if not queue:
                self.time_in_phase[t] += 1
                continue
            queue[0][1] -= 1
            if queue[0][1] > 0:
                continue
            queue.pop(0)
            if queue:
                self.api.trafficlight.setRedYellowGreenState(t, queue[0][0])
            else:  # transition finished: the new green starts
                self.phase[t] = self.target[t]
                self.time_in_phase[t] = 0.0
                self.api.trafficlight.setRedYellowGreenState(t, self.junctions[t].green_states[self.phase[t]])

    # ------------------------------------------------------------------ observation
    def states(self) -> dict[str, JunctionState]:
        return {t: self._state(j) for t, j in self.junctions.items()}

    def _lane_state(self, lane: str) -> LaneState:
        L = self.api.lane
        return LaneState(lane, L.getLastStepHaltingNumber(lane), L.getLastStepVehicleNumber(lane),
                         L.getWaitingTime(lane), min(1.0, L.getLastStepOccupancy(lane)))

    def _state(self, j: Junction) -> JunctionState:
        return JunctionState(
            tls_id=j.tls_id,
            time=self.time(),
            num_phases=len(j.green_states),
            phase=self.phase[j.tls_id],
            time_in_phase=self.time_in_phase[j.tls_id],
            in_transition=bool(self.transition[j.tls_id]),
            lanes=tuple(self._lane_state(lane) for lane in j.in_lanes),
            out_lanes=tuple(self._lane_state(lane) for lane in j.out_lanes),
            special_vehicles=self._special_vehicles(j),
            crossings=tuple(self._crossing(j, c) for c in j.crossings),
            phase_lanes=j.phase_lanes,
            phase_crossings=j.phase_crossings,
        )

    def _special_vehicles(self, j: Junction) -> tuple[SpecialVehicle, ...]:
        """Buses and emergency vehicles on the incoming lanes of this junction."""
        V, result = self.api.vehicle, []
        for lane in j.in_lanes:
            for v in self.api.lane.getLastStepVehicleIDs(lane):
                kind = V.getVehicleClass(v)
                if kind not in ("bus", "emergency"):
                    continue
                distance = self.api.lane.getLength(lane) - V.getLanePosition(v)
                # Our definition of "late" for a bus: time lost compared with driving at full speed.
                delay = V.getTimeLoss(v) if kind == "bus" else None
                result.append(SpecialVehicle(v, kind, lane, distance, V.getSpeed(v), delay))
        return tuple(result)

    def _crossing(self, j: Junction, crossing_id: str) -> Crossing:
        P = self.api.person
        waiting = [p for w in j.walking_areas for p in self.api.edge.getLastStepPersonIDs(w)
                   if P.getNextEdge(p) == crossing_id and P.getSpeed(p) < HALT_SPEED]
        max_wait = max((P.getWaitingTime(p) for p in waiting), default=0.0)
        on_crossing = len(self.api.edge.getLastStepPersonIDs(crossing_id))
        return Crossing(crossing_id, len(waiting), max_wait, on_crossing)

    def movements(self, tls_id: str) -> dict[int, list[tuple[str, str]]]:
        """For MaxPressure: {green phase: [(incoming lane, outgoing lane), ...]}."""
        j = self.junctions[tls_id]
        return {p: [(link[0][0], link[0][1]) for i, link in enumerate(j.links)
                    if link and is_green(state[i]) and not j._is_crossing(link)]
                for p, state in enumerate(j.green_states)}

    def emergency_approaches(self) -> list[tuple[str, str, float, float, str]]:
        """For the coordinator: (vehicle_id, tls_id, distance m, speed m/s, incoming lane)
        for every emergency vehicle and every traffic light still ahead of it."""
        V, result = self.api.vehicle, []
        for v in V.getIDList():
            if V.getVehicleClass(v) != "emergency":
                continue
            for tls_id, link_index, distance, _ in V.getNextTLS(v):
                lane = self.junctions[tls_id].links[link_index][0][0]
                result.append((v, tls_id, distance, V.getSpeed(v), lane))
        return result

    # ------------------------------------------------------------------ overlay (GUI only)
    def show(self, tls_id: str, label: str, color: list, warning: str = ""):
        """Draw two text labels near a junction in sumo-gui: who is in control and why,
        and the coordinator's warning. Does nothing without the GUI."""
        if not self.gui:
            return
        x, y = self.api.junction.getPosition(tls_id)
        shift = 40 + 45 * (list(self.junctions).index(tls_id) % 2)  # every other junction higher: no overlap
        for name, text, dy, rgba in (("label", label, shift, color), ("warning", warning, -shift, [220, 0, 0, 255])):
            poi = f"{tls_id}_{name}"
            if poi not in self.pois:
                self.api.poi.add(poi, x + 15, y + dy, tuple(rgba), " ", 100, "", 8, 8)
                self.pois.add(poi)
            self.api.poi.setType(poi, text or " ")  # sumo-gui shows the POI type as text
            self.api.poi.setColor(poi, tuple(rgba) if text else (0, 0, 0, 0))

    # ------------------------------------------------------------------ totals
    def _count(self):
        """Called every second: simple totals for the summary (full KPIs: kpis.py)."""
        tot, sim = self.totals, self.api.simulation
        tot["waiting_veh_s"] += sum(self.api.lane.getLastStepHaltingNumber(lane)
                                    for j in self.junctions.values() for lane in j.in_lanes)
        tot["arrived"] += sim.getArrivedNumber()
        for v in sim.getDepartedIDList():
            if self.api.vehicle.getVehicleClass(v) == "emergency":
                tot["emergency_depart"][v] = self.time()
        for v in sim.getArrivedIDList():
            if v in tot["emergency_depart"]:
                tot["emergency_travel"][v] = self.time() - tot["emergency_depart"][v]

    def summary(self) -> dict:
        tot = self.totals
        return {
            "arrived_vehicles": tot["arrived"],
            "total_waiting_veh_s": round(tot["waiting_veh_s"], 1),
            "mean_waiting_per_vehicle_s": round(tot["waiting_veh_s"] / max(tot["arrived"], 1), 2),
            "emergency_travel_s": dict(tot["emergency_travel"]),
        }
