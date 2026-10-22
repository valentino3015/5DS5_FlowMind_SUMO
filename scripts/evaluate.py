"""Evaluation benchmark: compares controllers head-to-head across seeds and KPIs.

Usage:
    python scripts/evaluate.py [--scenario 3x3grid] [--controllers fixed_time flow] [--seconds 600] [--seeds 100 101]

Owner: Evaluation team / Ahmed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmind.agents.baselines import FixedTime, MaxPressure
from flowmind.agents.flow import FlowAgent
from flowmind.env import SumoCorridorEnv
from flowmind.kpis import SimulationKPIs, print_kpi_report


def load_config() -> dict:
    config_path = ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single_simulation(
    controller: str,
    scenario: str,
    seconds: int,
    seed: int,
    config: dict,
) -> SimulationKPIs:
    """Run one simulation and return computed KPIs."""
    cfg = dict(config)
    cfg.setdefault("sim", {})["scenario"] = scenario
    cfg["sim"]["seconds"] = seconds

    if scenario == "3x3grid":
        cfg["sim"]["net_file"] = "scenarios/3x3grid/3x3Grid2lanes.net.xml"
        cfg["sim"]["route_file"] = "scenarios/3x3grid/routes14000.rou.xml"
    elif scenario == "cologne3":
        cfg["sim"]["net_file"] = "scenarios/cologne3.net.xml"
        cfg["sim"]["route_file"] = "scenarios/cologne3.rou.xml"

    env = SumoCorridorEnv(config=cfg, use_gui=False, seed=seed)
    states = env.reset()

    # Instantiate controller
    agents = {}
    for tls in env.tls_ids:
        if controller == "fixed_time":
            agents[tls] = FixedTime(cfg.get("fixed_time", {}))
        elif controller == "max_pressure":
            movements = env.get_phase_movements(tls)
            agents[tls] = MaxPressure(cfg.get("max_pressure", {}), movements)
        elif controller == "flow":
            agents[tls] = FlowAgent(cfg.get("flow", {}))
        elif controller == "flow_heuristic":
            flow_cfg = dict(cfg.get("flow", {}))
            flow_cfg.pop("model_path", None)
            agents[tls] = FlowAgent(flow_cfg)

    done = False
    try:
        while not done:
            actions = {}
            for tls, st in states.items():
                allowed = [True] * st.num_phases
                actions[tls] = agents[tls].act(st, [], allowed)
            states, done, _ = env.step(actions)
            sim_time = states[env.tls_ids[0]].time
            if sim_time - env.begin_time >= seconds:
                break

        kpis = env.get_kpis(controller=controller)
        return kpis

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate traffic controllers head-to-head")
    parser.add_argument("--scenario", default="3x3grid", choices=["3x3grid", "cologne3"],
                        help="Scenario to evaluate (default: 3x3grid)")
    parser.add_argument("--controllers", nargs="+", default=["fixed_time", "flow_heuristic", "flow"],
                        choices=["fixed_time", "max_pressure", "flow", "flow_heuristic"],
                        help="Controllers to evaluate")
    parser.add_argument("--seconds", type=int, default=600, help="Duration per run in seconds")
    parser.add_argument("--seeds", nargs="+", type=int, default=[100], help="Random seeds to evaluate")
    args = parser.parse_args()

    config = load_config()

    print("================================================================")
    print("                FlowMind Comprehensive Evaluation               ")
    print("================================================================")
    print(f"Scenario    : {args.scenario}")
    print(f"Controllers : {', '.join(args.controllers)}")
    print(f"Duration    : {args.seconds}s per simulation")
    print(f"Seeds       : {args.seeds}")
    print("----------------------------------------------------------------\n")

    records = []

    for ctrl in args.controllers:
        for seed in args.seeds:
            print(f"--> Running {ctrl.upper()} [Seed {seed}, {args.seconds}s]...")
            kpis = run_single_simulation(ctrl, args.scenario, args.seconds, seed, config)
            records.append({
                "Controller": ctrl,
                "Seed": seed,
                "Arrived (vehs)": kpis.total_arrived,
                "Total Departed": kpis.total_departed,
                "Arrival Rate (%)": round(kpis.arrival_rate, 1),
                "Avg Travel Time (s)": round(kpis.average_travel_time, 2),
                "Avg Wait Time (s)": round(kpis.average_waiting_time, 2),
                "Total Wait Time (s)": round(kpis.total_waiting_time, 1),
                "Mean Speed (m/s)": round(kpis.average_speed, 2),
                "Mean Speed (km/h)": round(kpis.average_speed_kmh, 1),
                "Cumulative Halting (veh*s)": round(kpis.cumulative_halting, 1),
                "Box Incidents": kpis.total_box_vehicles,
            })

    df = pd.DataFrame(records)

    # Output CSV to runs/
    output_dir = ROOT / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"evaluation_{args.scenario}.csv"
    df.to_csv(csv_path, index=False)

    print("\n================================================================")
    print("                 HEAD-TO-HEAD KPI BENCHMARK                     ")
    print("================================================================")

    # Group by controller and compute mean across seeds
    numeric_cols = [c for c in df.columns if c not in ["Controller", "Seed"]]
    summary = df.groupby("Controller")[numeric_cols].mean().reset_index()

    display_cols = [
        "Controller",
        "Arrived (vehs)",
        "Avg Travel Time (s)",
        "Avg Wait Time (s)",
        "Mean Speed (m/s)",
        "Cumulative Halting (veh*s)",
        "Box Incidents"
    ]
    print(summary[display_cols].to_string(index=False))
    print("================================================================")
    print(f"Full benchmark CSV saved to: {csv_path}\n")


if __name__ == "__main__":
    main()
