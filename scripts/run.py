"""Run one simulation with FlowMind.

Usage:
    python scripts/run.py --controller fixed_time [--gui] [--seed 100]
    python scripts/run.py --controller max_pressure [--gui]
    python scripts/run.py --controller flow [--gui]

Owner: Ahmed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmind.agents.baselines import FixedTime, MaxPressure
from flowmind.agents.flow import FlowAgent
from flowmind.agents.rules import RuleBasedAgent
from flowmind.env import SumoCorridorEnv


def load_config() -> dict:
    config_path = ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run one FlowMind simulation")
    parser.add_argument(
        "--controller",
        default="fixed_time",
        choices=["fixed_time", "rules", "max_pressure", "flow", "orchestrated"],
        help="Controller to run (default: fixed_time)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        choices=["3x3grid", "cologne3"],
        help="Traffic scenario: 3x3grid (default) or cologne3",
    )
    parser.add_argument("--gui", action="store_true", help="Open SUMO GUI")
    parser.add_argument("--heuristic", action="store_true", help="Run FlowAgent heuristic directly without loading checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Simulation seed")
    parser.add_argument("--seconds", type=int, default=600, help="Simulation duration (s)")
    args = parser.parse_args()

    config = load_config()
    if args.scenario:
        config.setdefault("sim", {})["scenario"] = args.scenario
        if args.scenario == "3x3grid":
            config["sim"]["net_file"] = "scenarios/3x3grid/3x3Grid2lanes.net.xml"
            config["sim"]["route_file"] = "scenarios/3x3grid/routes14000.rou.xml"
        elif args.scenario == "cologne3":
            config["sim"]["net_file"] = "scenarios/cologne3.net.xml"
            config["sim"]["route_file"] = "scenarios/cologne3.rou.xml"

    if args.seconds:
        config.setdefault("sim", {})["seconds"] = args.seconds

    active_scenario = config.get("sim", {}).get("scenario", "3x3grid")
    print(f"== FlowMind Simulation Runner ==")
    print(f"Scenario  : {active_scenario}")
    print(f"Controller: {args.controller} | GUI: {args.gui} | Seed: {args.seed} | Duration: {args.seconds}s\n")

    env = SumoCorridorEnv(config=config, use_gui=args.gui, seed=args.seed)
    states = env.reset()

    # Instantiate agents for each traffic light
    agents = {}
    for tls in env.tls_ids:
        if args.controller == "fixed_time":
            agents[tls] = FixedTime(config.get("fixed_time", {}))
        elif args.controller == "max_pressure":
            movements = env.get_phase_movements(tls)
            agents[tls] = MaxPressure(config.get("max_pressure", {}), movements)
        elif args.controller == "rules":
            agents[tls] = RuleBasedAgent(config.get("rules", {}) | {
                "min_green": config["signals"]["min_green"],
                "max_green": config["signals"]["max_green"],
            })
        elif args.controller == "flow":
            flow_cfg = dict(config.get("flow", {}))
            if args.heuristic:
                flow_cfg.pop("model_path", None)
            agents[tls] = FlowAgent(flow_cfg)
        elif args.controller == "orchestrated":
            # Fallback until orchestrator is delivered
            print(f"Note: Orchestrator is in development. Running flow agent for {tls}.")
            agents[tls] = FlowAgent(config.get("flow", {}))

    step_count = 0
    done = False

    try:
        while not done:
            actions = {}
            for tls, st in states.items():
                allowed = [True] * st.num_phases
                actions[tls] = agents[tls].act(st, [], allowed)

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


if __name__ == "__main__":
    main()
