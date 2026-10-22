"""Evaluate the incident agent: PPO response vs rule-based baseline vs no agent.

Usage:
    python scripts/evaluate_incident.py                       (all 3 controllers)
    python scripts/evaluate_incident.py --controllers rules ppo
    python scripts/evaluate_incident.py --controllers fixed_time rules --seeds 1   (quick check)

Every controller faces exactly the same breakdowns: for each evaluation seed
(incident.eval_seeds, never used in training) the same road is blocked at the
same time, drawn like in training.

Controllers:
    fixed_time : no incident agent, the fixed timer everywhere (today's lights)
    rules      : incident agent with the rule-based response (the baseline)
    ppo        : incident agent with the trained PPO response

KPIs, measured from the breakdown until 2 min after the car leaves:
    waiting_free   : vehicles waiting at the blocked junction on lanes that CAN move (PPO's reward)
    halting_all    : vehicles halting in the whole corridor
    mean_speed     : mean speed of all vehicles (m/s)
    detect_s       : seconds from the breakdown to the first incident claim
    unsafe_choices : PPO picks the rule had to replace (should be 0 after training)

Results: runs/incident_eval/results.csv, and a table with mean +- 95% CI.

Owner: Donia.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmind.agents.baselines import FixedTime
from flowmind.agents.incident import IncidentAgent
from flowmind.env import SumoCorridorEnv
from scripts.demo_incident import SCENARIO, load_config, read_scenario
from scripts.train_incident import incident_roads, write_block_file

OUT_DIR = ROOT / "runs" / "incident_eval"
KPIS = ["waiting_free", "halting_all", "mean_speed", "detect_s", "unsafe_choices"]


def draw_breakdown(roads: list, train: dict, begin: float, seed: int) -> tuple:
    """Same road / start / end for every controller with this seed."""
    rng = np.random.default_rng(seed)
    edge, lanes, length = roads[rng.integers(len(roads))]
    start = begin + rng.uniform(*train["block_start"])
    end = start + rng.uniform(*train["block_duration"])
    return edge, lanes, length, start, end


def run_episode(config: dict, controller: str, seed: int, model) -> dict:
    params = dict(config["incident"], use_ppo=False)
    train = params["train"]
    sc = read_scenario(SCENARIO)
    base = [r for r in sc["routes"] if not r.name.endswith(".block.xml")]
    edge, lanes, length, start, end = draw_breakdown(
        incident_roads(sc["net"], train["min_road_length"]), train, sc["begin"], seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    block_file = OUT_DIR / "block.xml"
    write_block_file(block_file, edge, lanes, length, start, end)
    cfg = dict(config, sim=dict(config["sim"], seconds=int(end - sc["begin"] + train["after_block"])))
    env = SumoCorridorEnv(config=cfg, net_file=sc["net"],
                          route_file=",".join(str(r) for r in base + [block_file]),
                          use_gui=False, seed=seed, begin_time=sc["begin"])
    states = env.reset()
    blocked_tls = next(t for t, st in states.items() if any(l.lane_id.startswith(edge + "_") for l in st.lanes))
    agents = {t: IncidentAgent(params, env.get_phase_movements(t), model if controller == "ppo" else None)
              for t in env.tls_ids}
    timers = {t: FixedTime(config["fixed_time"]) for t in env.tls_ids}

    waiting, halting, speed, first_claim, unsafe = [], [], [], None, 0
    done = False
    try:
        while not done:
            actions = {}
            for tls, st in states.items():
                allowed = [True] * st.num_phases
                claim = agents[tls].wants_control(st, [])  # detector runs for every controller
                if claim and controller != "fixed_time":
                    actions[tls] = agents[tls].act(st, [], allowed)
                    unsafe += agents[tls].last_reason.startswith("PPO chose unsafe")
                else:
                    actions[tls] = timers[tls].act(st, [], allowed)
                if claim and tls == blocked_tls and first_claim is None and st.time >= start:
                    first_claim = st.time
            states, done, info = env.step(actions)
            if info["time"] >= start:
                st = states[blocked_tls]
                free = [l.queue for l in st.lanes if not l.lane_id.startswith(edge + "_")]
                waiting.append(sum(free))
                halting.append(info["halting_vehicles"])
                speed.append(info["mean_speed"])
    finally:
        env.close()

    return {
        "controller": controller, "seed": seed, "road": edge, "junction": blocked_tls,
        "waiting_free": float(np.mean(waiting)), "halting_all": float(np.mean(halting)),
        "mean_speed": float(np.mean(speed)),
        "detect_s": (first_claim - start) if first_claim else float("nan"),
        "unsafe_choices": unsafe,
    }


def mean_ci(values: list[float]) -> str:
    v = np.array([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return "n/a"
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return f"{v.mean():7.2f} +- {half:5.2f}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate the incident agent")
    parser.add_argument("--controllers", nargs="+", default=["fixed_time", "rules", "ppo"],
                        choices=["fixed_time", "rules", "ppo"])
    parser.add_argument("--model", default=None, help="PPO model (default: incident.model_path)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Override incident.eval_seeds")
    args = parser.parse_args()

    config = load_config()
    seeds = args.seeds or [int(s) for s in config["incident"]["eval_seeds"]]
    model = None
    if "ppo" in args.controllers:
        from stable_baselines3 import PPO
        model = PPO.load(str(ROOT / (args.model or config["incident"]["model_path"])))

    rows = []
    for controller in args.controllers:
        for seed in seeds:
            row = run_episode(config, controller, seed, model)
            rows.append(row)
            print(f"{controller:10s} seed {seed:3d}  {row['junction'][:12]:12s} "
                  f"waiting_free {row['waiting_free']:6.2f}  halting_all {row['halting_all']:6.1f}  "
                  f"detect {row['detect_s']:5.0f}s  unsafe {row['unsafe_choices']}")

    with open(OUT_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n== Mean +- 95% CI over {len(seeds)} evaluation seeds (lower is better, except mean_speed) ==")
    print(f"{'controller':12s}" + "".join(f"{k:>18s}" for k in KPIS))
    for controller in args.controllers:
        mine = [r for r in rows if r["controller"] == controller]
        print(f"{controller:12s}" + "".join(f"{mean_ci([r[k] for r in mine]):>18s}" for k in KPIS))
    print(f"\nDetails: {OUT_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
