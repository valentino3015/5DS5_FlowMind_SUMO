"""Demo of the incident agent on the incident scenario.

Usage:
    python scripts/demo_incident.py --gui
    python scripts/demo_incident.py            (no window, text only)

Scenario: scenarios/cologne3_incident.sumocfg. A red car breaks down on one
approach of the first junction and blocks its lane for a while.

At the blocked junction a mini-orchestrator runs: if the incident agent claims
control it drives the light, otherwise the fixed timer does. The other
junctions run the fixed timer. The terminal prints what the agent sees and
when it takes and gives back control.

Owner: Donia. Demo only: the real orchestrator (control/orchestrator.py) will
replace the mini-orchestrator once it is delivered.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmind.agents.baselines import FixedTime
from flowmind.agents.incident import IncidentAgent
from flowmind.env import SumoCorridorEnv

SCENARIO = ROOT / "scenarios" / "cologne3_incident.sumocfg"


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_scenario(sumocfg: Path) -> dict:
    """Read net, routes and begin time from the .sumocfg, and the blockage from its .block.xml."""
    cfg = ET.parse(sumocfg).getroot()
    folder = sumocfg.parent
    routes = [folder / r for r in cfg.find("input/route-files").get("value").split(",")]
    block_file = next(r for r in routes if r.name.endswith(".block.xml"))
    car = ET.parse(block_file).getroot().find("vehicle")
    stop = car.find("stop")
    return {
        "net": folder / cfg.find("input/net-file").get("value"),
        "routes": routes,
        "begin": int(float(cfg.find("time/begin").get("value"))),
        "lane": stop.get("lane"),
        "block_start": float(car.get("depart")),
        "block_end": float(stop.get("until")),
    }


def find_junction(states: dict, lane_id: str) -> str:
    """The junction whose incoming lanes include the blocked lane."""
    for tls, st in states.items():
        if any(l.lane_id == lane_id for l in st.lanes):
            return tls
    raise ValueError(f"no junction controls lane {lane_id}")


def lane_info(state, lane_id: str) -> str:
    lane = next(l for l in state.lanes if l.lane_id == lane_id)
    green = lane_id in state.phase_lanes[state.phase]
    light = "GREEN" if green else "red  "
    return f"queue {lane.queue:2d} | occupancy {lane.occupancy:.2f} | light {light}"


def main():
    parser = argparse.ArgumentParser(description="Incident agent demo")
    parser.add_argument("--gui", action="store_true", help="Open SUMO GUI")
    parser.add_argument("--seed", type=int, default=None, help="Simulation seed")
    parser.add_argument("--seconds", type=int, default=900, help="Simulation duration (s)")
    args = parser.parse_args()

    config = load_config()
    config.setdefault("sim", {})["seconds"] = args.seconds
    sc = read_scenario(SCENARIO)
    block_end = sc["block_end"]

    env = SumoCorridorEnv(
        config=config,
        net_file=sc["net"],
        route_file=",".join(str(r) for r in sc["routes"]),  # SUMO accepts a comma list
        use_gui=args.gui,
        seed=args.seed,
        begin_time=sc["begin"],
    )
    states = env.reset()
    junction = find_junction(states, sc["lane"])
    timers = {tls: FixedTime(config["fixed_time"]) for tls in env.tls_ids}
    incident = IncidentAgent(config["incident"], env.get_phase_movements(junction))

    print("== FlowMind incident demo ==")
    print(f"Junction {junction}, lane {sc['lane']} blocked from t={sc['block_start']:.0f} "
          f"to t={block_end:.0f}\n")

    driver, detected_at, released_at, false_alarms = "fixed_time", None, None, 0
    done, next_print = False, 0.0
    try:
        while not done:
            actions = {}
            for tls, st in states.items():
                allowed = [True] * st.num_phases
                if tls != junction:
                    actions[tls] = timers[tls].act(st, [], allowed)
                    continue
                claim = incident.wants_control(st, [])
                new_driver = "incident" if claim else "fixed_time"
                agent = incident if claim else timers[tls]
                actions[tls] = agent.act(st, [], allowed)

                t = st.time
                blocked_now = sc["block_start"] <= t <= block_end + 60  # queue needs time to clear
                if sc["block_start"] <= t <= block_end + 120 and t >= next_print:
                    print(f"t={t:6.0f}  {lane_info(st, sc['lane'])}  | control: {driver}")
                    next_print = t + 15
                if new_driver != driver:
                    if claim:
                        print(f"\n>>> t={t:.0f}  INCIDENT AGENT TAKES CONTROL: {claim.reason}")
                        print(f"    action: {agent.explain()}\n")
                        if blocked_now and detected_at is None:
                            detected_at = t
                        elif not blocked_now:
                            false_alarms += 1
                    else:
                        print(f"\n<<< t={t:.0f}  incident agent lets go (lane moving again, "
                              f"or recheck: the lane gets a green to test it)\n")
                        released_at = t
                    driver = new_driver

            states, done, info = env.step(actions)
    finally:
        env.close()

    print("\n== Summary ==")
    if detected_at is None:
        print("Blockage NOT detected.")
    else:
        print(f"Detected {detected_at - sc['block_start']:.0f} s after the car appeared "
              f"(it first waits in the queue before reaching its spot).")
    if released_at is not None:
        print(f"Last claim ended at t={released_at:.0f} (car left at t={block_end:.0f}).")
    print(f"False alarms (claims outside the blockage): {false_alarms}")


if __name__ == "__main__":
    main()
