"""Run one simulation with FlowMind.

Usage:
    python scripts/run.py --controller fixed_time [--gui] [--seed 100]
    python scripts/run.py --controller green_wave --scenario corridor   (static green wave)
    python scripts/run.py --controller max_pressure [--gui]
    python scripts/run.py --controller flow [--gui]
    python scripts/run.py --controller orchestrated --scenario corridor [--gui]
    python scripts/run.py --controller orchestrated --scenario corridor --override J2:2:300:30
        (operator forces phase 2 at J2 from t=300 s for 30 s; the shield still applies)

--scenario corridor is the 3-junction street with buses, an ambulance and
pedestrians (scenarios/corridor/). The others are 3x3grid and cologne3.

Owner: Ahmed (runner, controllers). Orchestrated mode, decision log: Adem.
Results go to runs/<date>_<controller>_<scenario>_seed<N>/ (decisions.csv, summary.csv).
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import yaml

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmind.agents.baselines import FixedTime, GreenWave, MaxPressure, green_wave_offset
from flowmind.agents.bus import BusAgent
from flowmind.agents.demo_rules import DemoEmergency
from flowmind.agents.emergency import EmergencyAgent
from flowmind.agents.flow import FlowAgent
from flowmind.agents.incident import IncidentAgent
from flowmind.agents.pedestrian import PedestrianAgent
from flowmind.control import shield
from flowmind.control.coordinator import Coordinator
from flowmind.control.orchestrator import Decision, Orchestrator
from flowmind.env import SumoCorridorEnv
from flowmind.logger import DecisionLogger, summarize_tripinfo


def load_config() -> dict:
    config_path = ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_orchestrator(config: dict, movements: dict) -> Orchestrator:
    """One orchestrator (with its own agents) for one junction."""
    emergency = EmergencyAgent if config.get("agents", {}).get("emergency") == "team" else DemoEmergency
    agents = [
        emergency(config["emergency"]),
        IncidentAgent(config["incident"]),
        PedestrianAgent(config["pedestrian"]),
        BusAgent(config["bus"]),
        FlowAgent(config.get("flow", {})),
    ]
    o = config["orchestrator"]
    return Orchestrator(agents, o["priority"], o["min_hold"],
                        fallback=MaxPressure(config.get("max_pressure", {}), movements),
                        shadow=FixedTime(config.get("fixed_time", {})))  # what-if column only


def parse_override(text: str) -> tuple[str, int, float, float]:
    """"J2:2:300:30" -> ("J2", 2, 300.0, 30.0): junction, phase, start time, duration."""
    tls, phase, start, duration = text.split(":")
    return tls, int(phase), float(start), float(duration)


def default_run_dir(controller: str, scenario: str, seed_text: str, pid: int | None = None) -> Path:
    """runs/<date-time>_<controller>_<scenario>_seed<N>_p<process id>.

    The process id keeps two runs started in the same second apart
    (otherwise both write the same tripinfo.xml and the second one crashes)."""
    pid = os.getpid() if pid is None else pid
    return ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{controller}_{scenario}_seed{seed_text}_p{pid}"


def warning_text(events: list) -> str:
    """Short text for the overlay, e.g. "WARNING amb_1 in 15 s"."""
    parts = []
    for e in events:
        if e.kind == "emergency":
            parts.append(f"{e.data['vehicle_id']} in {e.data['eta']:.0f} s")
        elif e.data.get("cause") == "spillback":
            parts.append(f"road to {e.data['from_tls']} {e.data['occupancy']:.0%} full")
        else:
            parts.append(f"incident at {e.data['from_tls']}")
    return "WARNING " + ", ".join(parts) if parts else ""


def main():
    parser = argparse.ArgumentParser(description="Run one FlowMind simulation")
    parser.add_argument(
        "--controller",
        default="fixed_time",
        choices=["fixed_time", "green_wave", "max_pressure", "flow", "orchestrated"],
        help="Controller to run (default: fixed_time)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        choices=["3x3grid", "cologne3", "corridor"],
        help="Traffic scenario: 3x3grid (default), cologne3 or corridor",
    )
    parser.add_argument("--gui", action="store_true", help="Open SUMO GUI")
    parser.add_argument("--seed", type=int, default=None, help="Simulation seed")
    parser.add_argument("--seconds", type=int, default=600, help="Simulation duration (s)")
    parser.add_argument("--override", action="append", default=[], metavar="TLS:PHASE:START:DURATION",
                        help="orchestrated only: operator forces a phase, e.g. J2:2:300:30 (repeatable)")
    parser.add_argument("--run-dir", default=None, help="folder for the results (default: runs/<date>_...)")
    args = parser.parse_args()

    config = load_config()
    begin_time = None  # None = start at the first departure in the route file
    if args.scenario:
        config.setdefault("sim", {})["scenario"] = args.scenario
        if args.scenario == "3x3grid":
            config["sim"]["net_file"] = "scenarios/3x3grid/3x3Grid2lanes.net.xml"
            config["sim"]["route_file"] = "scenarios/3x3grid/routes14000.rou.xml"
        elif args.scenario == "cologne3":
            config["sim"]["net_file"] = "scenarios/cologne3.net.xml"
            config["sim"]["route_file"] = "scenarios/cologne3.rou.xml"
        elif args.scenario == "corridor":
            config["sim"]["net_file"] = "scenarios/corridor/corridor.net.xml"
            config["sim"]["route_file"] = "scenarios/corridor/corridor.rou.xml"
            begin_time = 0  # traffic is made of flows that start at t = 0

    if args.seconds:
        config.setdefault("sim", {})["seconds"] = args.seconds

    active_scenario = config.get("sim", {}).get("scenario", "3x3grid")
    seed_text = args.seed if args.seed is not None else "default"
    print(f"== FlowMind Simulation Runner ==")
    print(f"Scenario  : {active_scenario}")
    print(f"Controller: {args.controller} | GUI: {args.gui} | Seed: {seed_text} | Duration: {args.seconds}s\n")

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.controller, active_scenario, seed_text)
    tripinfo = run_dir / "tripinfo.xml"
    logger = DecisionLogger(run_dir, latency_budget_ms=config["orchestrator"]["max_decision_ms"])

    env = SumoCorridorEnv(config=config, use_gui=args.gui, seed=args.seed,
                          begin_time=begin_time, tripinfo_file=tripinfo)
    states = env.reset()

    # Instantiate agents for each traffic light
    agents = {}
    orchestrators = {}
    for tls in env.tls_ids:
        if args.controller == "fixed_time":
            agents[tls] = FixedTime(config.get("fixed_time", {}))
        elif args.controller == "green_wave":
            order = config.get("corridor", {}).get("order", [])
            agents[tls] = GreenWave(config["green_wave"], green_wave_offset(config["green_wave"], order, tls))
        elif args.controller == "max_pressure":
            movements = env.get_phase_movements(tls)
            agents[tls] = MaxPressure(config.get("max_pressure", {}), movements)
        elif args.controller == "flow":
            agents[tls] = FlowAgent(config.get("flow", {}))
        elif args.controller == "orchestrated":
            orchestrators[tls] = make_orchestrator(config, env.get_phase_movements(tls))

    for text in args.override:  # operator overrides (core idea 9)
        tls, phase, start, duration = parse_override(text)
        if tls in orchestrators:
            orchestrators[tls].set_override(phase, start, duration)
            print(f"Operator override: {tls} phase {phase} from t={start:.0f} s for {duration:.0f} s")
        else:
            print(f"Override ignored: {tls} is not an orchestrated junction")

    # Corridor order for the coordinator (falls back to the junction list of the scenario)
    order = [t for t in config.get("corridor", {}).get("order", []) if t in env.tls_ids] or list(env.tls_ids)
    coordinator = Coordinator(config["coordinator"], order)
    overlay = config.get("overlay", {})

    step_count = 0
    done = False
    decisions: list[Decision] = []

    try:
        while not done:
            actions = {}
            now = next(iter(states.values())).time
            if args.controller == "orchestrated":
                events = coordinator.update(now, env.emergency_approaches(), decisions, states)
            else:
                events = {}
            decisions = []

            for tls, st in states.items():
                if args.controller == "orchestrated":
                    allowed = shield.allowed(st, config)
                    decision = orchestrators[tls].decide(st, events.get(tls, []), allowed)
                    actions[tls] = decision.final
                    color = overlay.get("colors", {}).get(decision.agent, overlay.get("default_color", [128, 128, 128, 255]))
                    env.show(tls, f"{tls} | {decision.agent.upper()}: {decision.explanation}", color,
                             warning_text(events.get(tls, [])))
                else:
                    allowed = [True] * st.num_phases
                    action = agents[tls].act(st, [], allowed)
                    actions[tls] = action
                    decision = Decision(tls, st.time, agents[tls].name, 0.0, "", action, action, "",
                                        agents[tls].explain())
                logger.log(decision, events.get(tls, []))
                decisions.append(decision)

            states, done, info = env.step(actions)
            step_count += 1

            if step_count % 20 == 0:
                print(
                    f"Step {step_count:4d} | Time: {info['time']:.0f}s | "
                    f"Active Cars: {info['vehicle_count']:3d} | "
                    f"Halting: {info['halting_vehicles']:3d} | "
                    f"Mean Speed: {info['mean_speed']:.2f} m/s"
                )

        print(f"\nSimulation finished successfully after {step_count} decision steps.")
        print(f"Final Status -> Active Vehicles: {info['vehicle_count']}, Halting: {info['halting_vehicles']}")

    finally:
        env.close()

    summary = summarize_tripinfo(tripinfo)
    summary.update(logger.stats())  # decision time and what-if figures
    logger.close(summary)
    print("\nSummary (trips = vehicles that finished their trip):")
    for key, value in summary.items():
        print(f"  {key:28s} {value}")
    shares = ", ".join(f"{a} {100 * n / sum(logger.control_steps.values()):.0f}%"
                       for a, n in logger.control_steps.most_common())
    print(f"  {'control share by agent':28s} {shares}")
    print(f"\nLog: {run_dir}")


if __name__ == "__main__":
    main()
