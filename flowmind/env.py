"""SUMO connection: the ONLY file that imports traci / libsumo.

Owner: Ahmed.

Starts SUMO, builds frozen JunctionState objects for each traffic light,
and executes phase changes with automatic yellow (3s) and all-red (2s) insertions.
"""
from __future__ import annotations

import os
import hashlib
from functools import lru_cache
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import yaml

import sumolib
import traci

from flowmind.contracts import Crossing, JunctionState, LaneState, SpecialVehicle, LaneEmissions

# Root directory of the repository
ROOT = Path(__file__).resolve().parent.parent

# The Euler integrator exhibited process-dependent crossing of an internal
# link in our Windows scenario (vehicle 22, second 79). Ballistic integration
# passed repeated full-window traces. This is a NEW simulation protocol:
# recompute both reference and agent results; never reuse old baselines.
SUMO_REPRO_OPTIONS = ("--random", "false", "--threads", "1",
                      "--step-length", "1", "--step-method.ballistic", "true")


@lru_cache(maxsize=4)
def _binary_hash(path, modified_ns, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simulation_protocol(gui=False):
    binary = Path(sumolib.checkBinary("sumo-gui" if gui else "sumo")).resolve()
    stat = binary.stat()
    return {"revision": "ballistic_v7", "options": list(SUMO_REPRO_OPTIONS),
            "binary": str(binary),
            "binary_sha256": _binary_hash(str(binary), stat.st_mtime_ns, stat.st_size)}


def _get_first_depart(route_file: str | Path) -> int:
    """Find the earliest departure time in the route file to synchronize simulation start."""
    try:
        for _, elem in ET.iterparse(str(route_file), events=("start",)):
            if elem.tag in ("vehicle", "trip"):
                return int(float(elem.attrib.get("depart", 0)))
    except Exception:
        pass
    return 0


class SumoCorridorEnv:
    """The central SUMO environment for the FlowMind traffic signal control system."""

    def __init__(
        self,
        config: dict | None = None,
        net_file: str | Path | None = None,
        route_file: str | Path | None = None,
        use_gui: bool | None = None,
        seed: int | None = None,
        begin_time: int | None = None,
    ):
        if config is None:
            config_path = ROOT / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        self.config = config

        sim_cfg = self.config.get("sim", {})
        sig_cfg = self.config.get("signals", {})

        self.delta_time: int = int(sim_cfg.get("delta_time", 5))
        self.sim_seconds: int = int(sim_cfg.get("seconds", 3600))
        self.use_gui: bool = bool(use_gui if use_gui is not None else sim_cfg.get("gui", False))
        self.seed: int = int(seed if seed is not None else self.config.get("evaluation", {}).get("seeds", [100])[0])

        self.yellow_time: int = int(sig_cfg.get("yellow", 3))
        self.all_red_time: int = int(sig_cfg.get("all_red", 2))
        self.max_red_clearance: int = int(sig_cfg.get("max_red_clearance", 4))
        self.min_green: int = int(sig_cfg.get("min_green", 10))
        self.max_green: int = int(sig_cfg.get("max_green", 60))

        # Discover or set scenario files
        scenarios_dir = ROOT / "scenarios"
        scenario_name = sim_cfg.get("scenario", "3x3grid")

        if net_file is None:
            if sim_cfg.get("net_file"):
                net_file = ROOT / sim_cfg["net_file"]
            elif scenario_name == "3x3grid":
                net_file = ROOT / "scenarios" / "3x3grid" / "3x3Grid2lanes.net.xml"
            elif scenario_name == "cologne3":
                net_file = ROOT / "scenarios" / "cologne3.net.xml"
            else:
                default_nets = list(scenarios_dir.glob("*.net.xml"))
                net_file = default_nets[0] if default_nets else ROOT / "scenarios" / "3x3grid" / "3x3Grid2lanes.net.xml"

        if route_file is None:
            if sim_cfg.get("route_file"):
                route_file = ROOT / sim_cfg["route_file"]
            elif scenario_name == "3x3grid":
                route_file = ROOT / "scenarios" / "3x3grid" / "routes14000.rou.xml"
            elif scenario_name == "cologne3":
                route_file = ROOT / "scenarios" / "cologne3.rou.xml"
            else:
                default_rou = list(scenarios_dir.glob("*.rou.xml"))
                route_file = default_rou[0] if default_rou else ROOT / "scenarios" / "3x3grid" / "routes14000.rou.xml"

        self.net_file = Path(net_file)
        self.route_file = Path(route_file)

        # Synchronize simulation start with earliest vehicle departure
        if begin_time is None:
            self.begin_time = _get_first_depart(self.route_file)
        else:
            self.begin_time = int(begin_time)

        self.is_running = False
        self.metrics_callback = None
        self.reproducible: bool = bool(sim_cfg.get("reproducible", False))
        self.tls_ids: list[str] = []
        self.green_states: dict[str, list[str]] = {}
        self.phase_lanes: dict[str, tuple[tuple[str, ...], ...]] = {}
        self.phase_crossings: dict[str, tuple[tuple[str, ...], ...]] = {}
        self.movements: dict[str, dict[int, list[tuple[str, str]]]] = {}
        # For each tls, map phase -> list of outgoing lane ids
        self.phase_out_lanes: dict[str, list[list[str]]] = {}
        # For each tls, tuple of internal lanes (":...") inside junction box
        self.junction_internal_lanes: dict[str, tuple[str, ...]] = {}

        self.current_phases: dict[str, int] = {}
        self.time_in_phase: dict[str, float] = {}
        self.in_transition: dict[str, bool] = {}

        # Topology caches (static data that does not change during simulation)
        self.cached_network_lanes: list[str] = []
        self.cached_internal_lanes: list[str] = []
        self.cached_controlled_lanes: dict[str, list[str]] = {}
        self.cached_out_lanes: dict[str, list[str]] = {}
        self.cached_lane_lengths: dict[str, float] = {}

        # Spillback threshold: don't release green into downstream lanes above this occupancy
        shield_cfg = self.config.get("shield", {})
        self.spillback_threshold: float = float(
            shield_cfg.get("spillback_threshold", 0.75)
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self) -> dict[str, JunctionState]:
        """Start or restart the SUMO simulation and return initial states."""
        if self.is_running:
            self.close()

        sumo_binary = sumolib.checkBinary("sumo-gui" if self.use_gui else "sumo")
        cmd = [
            sumo_binary,
            "-n", str(self.net_file),
            "-r", str(self.route_file),
            "-b", str(self.begin_time),
            "--seed", str(self.seed),
            "--time-to-teleport", "60",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
        ]

        # Opt-in (sim.reproducible, off by default): only the emission evaluation scripts use it.
        if self.reproducible:
            cmd.extend(SUMO_REPRO_OPTIONS)
        output_dir = self.config.get("evaluation_output")
        if output_dir:
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--tripinfo-output", str(output_dir / "tripinfo.xml"),
                    "--tripinfo-output.write-unfinished", "true",
                    "--device.emissions.probability", "1",
                    "--statistic-output", str(output_dir / "statistics.xml")]
        if self.use_gui and "gui_delay_ms" in self.config:  # set only by the dashboard
            cmd += ["--start", "--quit-on-end", "true",
                    "--delay", str(self.config.get("gui_delay_ms", 100))]
        traci.start(cmd)
        self.is_running = True
        self._init_traffic_lights()

        # Step once to populate initial vehicle/lane states
        traci.simulationStep()
        self._after_step()

        states = {tls: self.build_junction_state(tls) for tls in self.tls_ids}
        return states

    def reset(self) -> dict[str, JunctionState]:
        """Reset the environment to the beginning."""
        return self.start()

    def _past_horizon(self) -> bool:
        """Reproducible mode only: never simulate beyond sim.seconds. Always False otherwise."""
        return self.reproducible and traci.simulation.getTime() >= self.begin_time + self.sim_seconds

    def _after_step(self):
        """Sample every second (yellow/red included) when metrics_callback is set. No-op otherwise."""
        if self.metrics_callback:
            queues = [traci.lane.getLastStepHaltingNumber(lane) for lane in self.cached_network_lanes]
            vehicles = traci.vehicle.getIDList()
            self.metrics_callback({
                "elapsed_s": traci.simulation.getTime() - self.begin_time,
                "vehicles": len(vehicles),
                "departed": traci.simulation.getDepartedNumber(),
                "arrived": traci.simulation.getArrivedNumber(),
                "pending": len(traci.simulation.getPendingVehicles()),
                "halting": sum(queues),
                "queue_mean_lane": sum(queues) / len(queues) if queues else 0,
                "queue_max_lane": max(queues, default=0),
                "speed_sum_m_s": sum(traci.vehicle.getSpeed(v) for v in vehicles),
                "waiting_mean_active_s": sum(traci.vehicle.getWaitingTime(v) for v in vehicles) / len(vehicles) if vehicles else 0,
                "co2_mg_s": sum(traci.vehicle.getCO2Emission(v) for v in vehicles),
                "nox_mg_s": sum(traci.vehicle.getNOxEmission(v) for v in vehicles),
                "pmx_mg_s": sum(traci.vehicle.getPMxEmission(v) for v in vehicles),
                "fuel_mg_s": sum(traci.vehicle.getFuelConsumption(v) for v in vehicles),
                "collisions": traci.simulation.getCollidingVehiclesNumber(),
                "teleports": traci.simulation.getStartingTeleportNumber(),
            })

    def _init_traffic_lights(self):
        """Analyze traffic light logic, discovering green phases and controlled links."""
        self.tls_ids = list(traci.trafficlight.getIDList())
        self.green_states.clear()
        self.phase_lanes.clear()
        self.phase_crossings.clear()
        self.movements.clear()
        self.current_phases.clear()
        self.time_in_phase.clear()
        self.in_transition.clear()

        for tls in self.tls_ids:
            logics = traci.trafficlight.getAllProgramLogics(tls)
            phases = logics[0].phases if logics else []

            # A green phase contains 'G' or 'g' and no 'y'/'Y'
            green_indices = [
                i for i, p in enumerate(phases)
                if ("G" in p.state or "g" in p.state) and not ("y" in p.state or "Y" in p.state)
            ]
            if not green_indices:
                green_indices = list(range(len(phases)))

            self.green_states[tls] = [phases[idx].state for idx in green_indices]
            links = traci.trafficlight.getControlledLinks(tls)

            p_lanes: list[tuple[str, ...]] = []
            p_out_lanes: list[list[str]] = []
            tls_moves: dict[int, list[tuple[str, str]]] = {}

            for p_idx, state_str in enumerate(self.green_states[tls]):
                active_lanes = set()
                active_out = []
                moves = []
                for sig_idx, char in enumerate(state_str):
                    if char in ("G", "g") and sig_idx < len(links):
                        for link in links[sig_idx]:
                            src, dst = link[0], link[1]
                            active_lanes.add(src)
                            if not dst.startswith(":"):
                                moves.append((src, dst))
                                if dst not in active_out:
                                    active_out.append(dst)
                p_lanes.append(tuple(sorted(active_lanes)))
                p_out_lanes.append(active_out)
                tls_moves[p_idx] = moves

            self.phase_lanes[tls] = tuple(p_lanes)
            self.phase_out_lanes[tls] = p_out_lanes
            self.movements[tls] = tls_moves
            num_g = len(self.green_states[tls])
            self.phase_crossings[tls] = tuple(() for _ in range(num_g))

            # Cache static controlled lanes, unique outgoing lanes, and internal junction via-lanes
            ctrl = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls)))
            self.cached_controlled_lanes[tls] = ctrl
            out_ids = set()
            via_lanes = set()
            for link_list in links:
                for link in link_list:
                    to_lane = link[1]
                    if to_lane and not to_lane.startswith(":"):
                        out_ids.add(to_lane)
                    if len(link) > 2 and link[2] and link[2].startswith(":"):
                        via_lanes.add(link[2])
            self.cached_out_lanes[tls] = sorted(out_ids)
            self.junction_internal_lanes[tls] = tuple(sorted(via_lanes))

            self.current_phases[tls] = 0
            self.time_in_phase[tls] = 0.0
            self.in_transition[tls] = False

            # Set initial green phase
            traci.trafficlight.setRedYellowGreenState(tls, self.green_states[tls][0])

        # Cache global network and internal lanes + lane lengths
        all_lanes = traci.lane.getIDList()
        self.cached_network_lanes = [l for l in all_lanes if not l.startswith(":")]
        self.cached_internal_lanes = [l for l in all_lanes if l.startswith(":")]
        for l in all_lanes:
            try:
                self.cached_lane_lengths[l] = float(traci.lane.getLength(l))
            except Exception:
                self.cached_lane_lengths[l] = 50.0

    def is_box_occupied(self, tls: str) -> bool:
        """Return True if any vehicles are currently moving or stopped inside this junction box."""
        via_lanes = self.junction_internal_lanes.get(tls, ())
        return any(traci.lane.getLastStepVehicleNumber(l) > 0 for l in via_lanes)

    def get_box_halting_count(self, tls: str) -> int:
        """Return count of stopped/trapped vehicles inside this junction box."""
        via_lanes = self.junction_internal_lanes.get(tls, ())
        return sum(traci.lane.getLastStepHaltingNumber(l) for l in via_lanes)

    def get_phase_movements(self, tls_id: str) -> dict[int, list[tuple[str, str]]]:
        """Return movements for a junction, needed by the MaxPressure agent."""
        return self.movements.get(tls_id, {})

    def _create_yellow_state(self, current_state: str, next_state: str) -> str:
        """Create yellow transition state turning expiring greens into yellow."""
        res = []
        for c, n in zip(current_state, next_state):
            if c in ("G", "g") and n in ("r", "s"):
                res.append("y")
            elif c in ("G", "g"):
                res.append("G")
            else:
                res.append("r")
        return "".join(res)

    def _is_downstream_blocked(self, tls: str, target_phase: int) -> bool:
        """Return True if ALL outgoing lanes for this phase are severely jammed (> spillback_threshold)."""
        out_lanes = self.phase_out_lanes.get(tls, [])
        if not out_lanes or target_phase >= len(out_lanes) or not out_lanes[target_phase]:
            return False  # No downstream data -> allow

        occ_values = []
        for lane in out_lanes[target_phase]:
            try:
                occ_values.append(traci.lane.getLastStepOccupancy(lane))
            except Exception:
                pass
        if not occ_values:
            return False
        # Block only if EVERY exit lane is jam-packed (> spillback_threshold)
        return all(o >= self.spillback_threshold for o in occ_values)

    def step(self, actions: dict[str, int]) -> tuple[dict[str, JunctionState], bool, dict]:
        """Execute one decision step (delta_time seconds), inserting yellow/all-red if phases change."""
        step_start_time = traci.simulation.getTime()
        transitions_needed: dict[str, int] = {}
        suppressed_transitions: list[str] = []   # junctions where spillback blocked a switch

        for tls, target_phase in actions.items():
            if tls not in self.tls_ids:
                continue
            num_phases = len(self.green_states[tls])
            target_phase = max(0, min(num_phases - 1, target_phase))
            current = self.current_phases[tls]
            t_in_phase = self.time_in_phase[tls]
            # Enforce max_green: if green has been on for >= max_green, FORCE CYCLE to next phase!
            # Starvation prevention: cross-traffic (horizontal) MUST get green after max_green seconds!
            if t_in_phase >= self.max_green and target_phase == current:
                target_phase = (current + 1) % num_phases

            if target_phase != current:
                forced_by_max_green = (t_in_phase >= self.max_green)
                # Max-out overrides suppression so cross-traffic is never starved
                if not forced_by_max_green and self._is_downstream_blocked(tls, target_phase):
                    suppressed_transitions.append(tls)
                else:
                    transitions_needed[tls] = target_phase

        if transitions_needed:
            # 1. Apply Yellow Phase for signals.yellow seconds
            for tls, target_phase in transitions_needed.items():
                curr_s = self.green_states[tls][self.current_phases[tls]]
                next_s = self.green_states[tls][target_phase]
                y_state = self._create_yellow_state(curr_s, next_s)
                traci.trafficlight.setRedYellowGreenState(tls, y_state)
                self.in_transition[tls] = True

            for _ in range(self.yellow_time):
                if self._past_horizon():
                    break
                traci.simulationStep()
                self._after_step()

            # 2. Apply All-Red Phase for signals.all_red seconds
            for tls in transitions_needed:
                red_state = "r" * len(self.green_states[tls][0])
                traci.trafficlight.setRedYellowGreenState(tls, red_state)

            for _ in range(self.all_red_time):
                if self._past_horizon():
                    break
                traci.simulationStep()
                self._after_step()

            # 2b. DYNAMIC RED CLEARANCE EXTENSION:
            # If vehicles are STILL clearing or caught inside the intersection box,
            # hold all-red (up to max_red_clearance extra seconds) so they exit
            # BEFORE releasing green into cross-traffic!
            clearance_extra = 0
            while clearance_extra < self.max_red_clearance:
                if not any(self.is_box_occupied(tls) for tls in transitions_needed):
                    break
                if self._past_horizon():
                    break
                traci.simulationStep()
                self._after_step()
                clearance_extra += 1

            # 3. Apply New Green Phase
            for tls, target_phase in transitions_needed.items():
                self.current_phases[tls] = target_phase
                self.time_in_phase[tls] = 0.0
                self.in_transition[tls] = False
                traci.trafficlight.setRedYellowGreenState(tls, self.green_states[tls][target_phase])

            # Advance remainder of delta_time if any
            elapsed = self.yellow_time + self.all_red_time + clearance_extra
            remainder = max(0, self.delta_time - elapsed)
            for _ in range(remainder):
                if self._past_horizon():
                    break
                traci.simulationStep()
                self._after_step()
        else:
            # No transitions: keep all lights green for full delta_time
            for _ in range(self.delta_time):
                if self._past_horizon():
                    break
                traci.simulationStep()
                self._after_step()

        # Update time_in_phase for ALL junctions (regardless of transition status)
        for tls in self.tls_ids:
            if tls not in transitions_needed:
                self.time_in_phase[tls] += float(self.delta_time)
                if self.reproducible:  # count the real elapsed time (all-red extensions, horizon)
                    self.time_in_phase[tls] += traci.simulation.getTime() - step_start_time - self.delta_time
                self.in_transition[tls] = False

        # Gather new junction states
        states = {tls: self.build_junction_state(tls) for tls in self.tls_ids}

        sim_time = traci.simulation.getTime()
        done = (sim_time - self.begin_time >= self.sim_seconds) or (traci.simulation.getMinExpectedNumber() <= 0)

        # Count vehicles inside junction boxes using cached internal lane IDs
        box_vehicles = sum(traci.lane.getLastStepHaltingNumber(l) for l in self.cached_internal_lanes)

        # Collect basic metrics using cached network lane IDs
        halting = sum(traci.lane.getLastStepHaltingNumber(l) for l in self.cached_network_lanes)
        all_vehs = traci.vehicle.getIDList()
        vehicle_count = len(all_vehs)

        if vehicle_count > 0:
            # Sample up to 100 vehicles evenly across the fleet for fast, accurate mean speed
            stride = max(1, vehicle_count // 100)
            sampled_vehs = all_vehs[::stride]
            mean_speed = float(sum(traci.vehicle.getSpeed(v) for v in sampled_vehs) / len(sampled_vehs))
        else:
            mean_speed = 0.0

        info = {
            "time": sim_time,
            "halting_vehicles": halting,
            "box_vehicles": box_vehicles,           # Cars stuck in intersection box
            "suppressed_switches": len(suppressed_transitions),  # Times spillback blocked a switch
            "vehicle_count": vehicle_count,
            "mean_speed": mean_speed,
        }

        return states, done, info

    def build_junction_state(self, tls_id: str) -> JunctionState:
        """Query SUMO and construct a frozen JunctionState contract object."""
        sim_time = float(traci.simulation.getTime())
        controlled_lanes = self.cached_controlled_lanes.get(
            tls_id, list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        )

        # Build incoming lane states
        in_lanes = []
        for lane in controlled_lanes:
            in_lanes.append(
                LaneState(
                    lane_id=lane,
                    queue=int(traci.lane.getLastStepHaltingNumber(lane)),
                    vehicles=int(traci.lane.getLastStepVehicleNumber(lane)),
                    waiting_time=float(traci.lane.getWaitingTime(lane)),
                    occupancy=float(traci.lane.getLastStepOccupancy(lane)),
                )
            )

        # Build outgoing lane states from cached IDs
        out_lanes = []
        for out in self.cached_out_lanes.get(tls_id, []):
            out_lanes.append(
                LaneState(
                    lane_id=out,
                    queue=int(traci.lane.getLastStepHaltingNumber(out)),
                    vehicles=int(traci.lane.getLastStepVehicleNumber(out)),
                    waiting_time=float(traci.lane.getWaitingTime(out)),
                    occupancy=float(traci.lane.getLastStepOccupancy(out)),
                )
            )

        # Scan for special vehicles (buses and emergency vehicles) on approaching lanes
        specials = []
        for lane in controlled_lanes:
            vehs_on_lane = traci.lane.getLastStepVehicleIDs(lane)
            lane_len = self.cached_lane_lengths.get(lane, 50.0)
            for v in vehs_on_lane:
                vtype = traci.vehicle.getTypeID(v).lower()
                vclass = traci.vehicle.getVehicleClass(v).lower()

                is_emergency = "emergency" in vclass or "emergency" in vtype or "amb" in v or "fire" in v
                is_bus = "bus" in vclass or "bus" in vtype or "bus" in v

                if is_emergency or is_bus:
                    pos = traci.vehicle.getLanePosition(v)
                    dist = max(0.0, float(lane_len - pos))
                    spd = float(traci.vehicle.getSpeed(v))
                    delay = float(traci.vehicle.getWaitingTime(v)) if is_bus else None
                    specials.append(
                        SpecialVehicle(
                            vehicle_id=v,
                            kind="emergency" if is_emergency else "bus",
                            lane_id=lane,
                            distance=dist,
                            speed=spd,
                            delay=delay,
                        )
                    )

        num_phases = len(self.green_states[tls_id])
        curr_phase = self.current_phases[tls_id]

        return JunctionState(
            lane_emissions=tuple(LaneEmissions(
                lane, max(0.0, traci.lane.getCO2Emission(lane)),
                max(0.0, traci.lane.getNOxEmission(lane)),
                max(0.0, traci.lane.getFuelConsumption(lane)),
                max(0.0, traci.lane.getPMxEmission(lane)),
            ) for lane in controlled_lanes),
            tls_id=tls_id,
            time=sim_time,
            num_phases=num_phases,
            phase=curr_phase,
            time_in_phase=self.time_in_phase[tls_id],
            in_transition=self.in_transition[tls_id],
            lanes=tuple(in_lanes),
            out_lanes=tuple(out_lanes),
            special_vehicles=tuple(specials),
            crossings=self.phase_crossings[tls_id],
            phase_lanes=self.phase_lanes[tls_id],
            phase_crossings=self.phase_crossings[tls_id],
            phase_out_lanes=tuple(
                tuple(lanes) for lanes in self.phase_out_lanes.get(tls_id, [])
            ),
        )

    def close(self):
        """Safely disconnect from the TraCI server."""
        if self.is_running:
            try:
                traci.close()
            except Exception:
                pass
            finally:
                self.is_running = False
