"""Run one simulation of the corridor.

Owner: Adem.

    python scripts/run.py --controller fixed_time            # baseline: fixed timer
    python scripts/run.py --controller orchestrated --gui    # our system, in sumo-gui
    options: --seed N  --seconds S  --scenario NAME

Each step (sim.delta_time s), for each junction:
    state -> shield.allowed -> orchestrator.decide (agent + shield check) -> log -> apply
Results go to runs/<date>_<controller>_seed<N>/ (decisions.csv, summary.csv).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so "import flowmind" works when run as a script

from flowmind.agents.baselines import FixedTime, MaxPressure  # noqa: E402
from flowmind.agents.bus import BusAgent  # noqa: E402
from flowmind.agents.emergency import EmergencyAgent  # noqa: E402
from flowmind.agents.flow import FlowAgent  # noqa: E402
from flowmind.agents.incident import IncidentAgent  # noqa: E402
from flowmind.agents.pedestrian import PedestrianAgent  # noqa: E402
from flowmind.control import shield  # noqa: E402
from flowmind.control.orchestrator import Orchestrator  # noqa: E402
from flowmind.env import CorridorEnv  # noqa: E402
from flowmind.logger import DecisionLogger  # noqa: E402


def make_orchestrator(controller: str, config: dict, movements: dict) -> Orchestrator:
    """One orchestrator (with its own agents) for one junction."""
    if controller == "fixed_time":
        # Baseline: no specialists, the fixed timer always drives (same pipeline, same log).
        return Orchestrator([], [], 0, fallback=FixedTime(config["fixed_time"]))
    agents = [
        EmergencyAgent(config["emergency"]),
        IncidentAgent(config["incident"]),
        PedestrianAgent(config["pedestrian"]),
        BusAgent(config["bus"]),
        FlowAgent(config["flow"]),
    ]
    o = config["orchestrator"]
    return Orchestrator(agents, o["priority"], o["min_hold"],
                        fallback=MaxPressure(config["max_pressure"], movements))


def main():
    p = argparse.ArgumentParser(description="Run the FlowMind corridor once.")
    p.add_argument("--controller", choices=["fixed_time", "orchestrated"], default="orchestrated")
    p.add_argument("--gui", action="store_true", help="open sumo-gui")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=int, default=None, help="override sim.seconds")
    p.add_argument("--scenario", default=None, help="folder in scenarios/ (default: sim.scenario)")
    args = p.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    if args.seconds:
        config["sim"]["seconds"] = args.seconds
    scenario = args.scenario or config["sim"]["scenario"]

    env = CorridorEnv(config, scenario, seed=args.seed, gui=args.gui or config["sim"]["gui"])
    states = env.reset()
    orchestrators = {t: make_orchestrator(args.controller, config, env.movements(t)) for t in states}
    run_dir = ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{args.controller}_seed{args.seed}"
    logger = DecisionLogger(run_dir)
    print(f"== {args.controller} | scenario {scenario} | seed {args.seed} | {config['sim']['seconds']} s ==")

    events: dict[str, list] = {t: [] for t in states}  # coordinator warnings (step 5)
    while not env.done():
        actions = {}
        for tls_id, state in states.items():
            allowed = shield.allowed(state, config)
            decision = orchestrators[tls_id].decide(state, events[tls_id], allowed)
            logger.log(decision, events[tls_id])
            actions[tls_id] = decision.final
        states = env.step(actions)

    summary = env.summary()
    env.close()
    logger.close(summary)
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key:28s} {value}")
    print("  control share by agent:     " + ", ".join(
        f"{a} {100 * n / sum(logger.control_steps.values()):.0f}%" for a, n in logger.control_steps.most_common()))
    print(f"\nLog: {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
