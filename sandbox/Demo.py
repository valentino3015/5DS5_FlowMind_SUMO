"""
Demo: Fixed-time signals vs Q-learning agent on a single intersection (SUMO-RL).

Run from inside the cloned sumo-rl folder:

    python demo.py                 -> baseline + training + comparison plot (no GUI)
    python demo.py --gui           -> same, but opens sumo-gui (slower, for showing live)
    python demo.py --episodes 3 --seconds 1800   -> quicker run
"""
import argparse
import os

# Auto-set SUMO_HOME before sumo_rl is imported (pip install sumo or MSI install)
if "SUMO_HOME" not in os.environ:
    try:
        import sumo
        os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    except ImportError:
        for _p in (r"C:\Program Files (x86)\Sumo", r"C:\Program Files\Sumo"):
            if os.path.isdir(_p):
                os.environ["SUMO_HOME"] = _p
                break

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

import sys
import sumo_rl

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Networks ship inside the sumo_rl package (sumo_rl/nets/...), works for clone or pip install
_NETS = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "single-intersection")
NET = os.path.join(_NETS, "single-intersection.net.xml")
ROUTE = os.path.join(_NETS, "single-intersection.rou.xml")


def make_env(gui, seconds, fixed):
    return SumoEnvironment(
        net_file=NET,
        route_file=ROUTE,
        use_gui=gui,
        num_seconds=seconds,
        min_green=10,
        max_green=50,
        fixed_ts=fixed,
        single_agent=False,
        sumo_warnings=False,
        add_per_agent_info=False,
        additional_sumo_cmd="--no-step-log",
    )


def reset(env):
    res = env.reset()
    return res[0] if isinstance(res, tuple) else res


def run_fixed(gui, seconds):
    """Baseline: default fixed-time signal program, no learning."""
    env = make_env(gui, seconds, fixed=True)
    reset(env)
    rows = []
    done = {"__all__": False}
    while not done["__all__"]:
        _, _, done, info = env.step({})
        rows.append(
            {
                "step": info["step"],
                "waiting_time": info["system_mean_waiting_time"],
                "stopped": info["system_total_stopped"],
            }
        )
    env.close()
    return pd.DataFrame(rows)


def run_ql(gui, seconds, episodes, alpha=0.1, gamma=0.99):
    """Independent Q-learning agent per traffic light. Returns metrics per episode."""
    env = make_env(gui, seconds, fixed=False)
    all_runs = []
    agents = None

    for ep in range(1, episodes + 1):
        states = reset(env)
        if agents is None:
            agents = {
                #ts = traffic signal
                ts: QLAgent(
                    starting_state=env.encode(states[ts], ts),
                    state_space=env.observation_space,
                    action_space=env.action_space,
                    alpha=alpha,
                    gamma=gamma,
                    exploration_strategy=EpsilonGreedy(
                        initial_epsilon=0.05, min_epsilon=0.005, decay=1.0
                    ),
                )
                for ts in env.ts_ids
            }
        else:
            for ts in env.ts_ids:
                agents[ts].state = env.encode(states[ts], ts)
                agents[ts].action = None

        rows = []
        done = {"__all__": False}
        while not done["__all__"]:
            actions = {ts: agents[ts].act() for ts in agents}
            s, r, done, info = env.step(action=actions)
            for ts in s:
                agents[ts].learn(next_state=env.encode(s[ts], ts), reward=r[ts])
            rows.append(
                {
                    "step": info["step"],
                    "waiting_time": info["system_mean_waiting_time"],
                    "stopped": info["system_total_stopped"],
                }
            )
        df = pd.DataFrame(rows)
        all_runs.append(df)
        print(f"[QL] episode {ep}/{episodes}  mean waiting time = {df['waiting_time'].mean():.2f}s")
    env.close()
    return all_runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gui", action="store_true", help="open sumo-gui")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seconds", type=int, default=3600)
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("outputs", exist_ok=True)

    print("== Fixed-time baseline ==")
    fixed = run_fixed(args.gui, args.seconds)
    print(f"[Fixed] mean waiting time = {fixed['waiting_time'].mean():.2f}s")
    fixed.to_csv("outputs/demo_fixed.csv", index=False)

    print("== Q-learning agent ==")
    runs = run_ql(args.gui, args.seconds, args.episodes)
    ql = runs[-1]
    ql.to_csv("outputs/demo_ql_last_episode.csv", index=False)

    # Plot 1: waiting time over the simulation
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(fixed["step"], fixed["waiting_time"], label="Fixed-time", color="tab:red")
    ax[0].plot(ql["step"], ql["waiting_time"], label="Q-learning (last episode)", color="tab:green")
    ax[0].set_xlabel("Simulation step (s)")
    ax[0].set_ylabel("Mean waiting time (s)")
    ax[0].set_title("Waiting time during the simulation")
    ax[0].legend()

    # Plot 2: summary bars
    f_mean = fixed["waiting_time"].mean()
    q_mean = ql["waiting_time"].mean()
    ax[1].bar(["Fixed-time", "Q-learning"], [f_mean, q_mean], color=["tab:red", "tab:green"])
    ax[1].set_ylabel("Average mean waiting time (s)")
    gain = 100 * (f_mean - q_mean) / f_mean if f_mean else 0
    ax[1].set_title(f"Average waiting time ({gain:+.1f}% vs baseline)")

    plt.tight_layout()
    plt.savefig("outputs/comparison.png", dpi=150)
    print("\nSaved: outputs/comparison.png, outputs/demo_fixed.csv, outputs/demo_ql_last_episode.csv")
    print(f"Fixed-time: {f_mean:.2f}s | Q-learning: {q_mean:.2f}s | improvement: {gain:.1f}%")

    # Training curve across episodes
    curve = [r["waiting_time"].mean() for r in runs]
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(curve) + 1), curve, marker="o")
    plt.axhline(f_mean, color="tab:red", linestyle="--", label="Fixed-time baseline")
    plt.xlabel("Episode")
    plt.ylabel("Mean waiting time (s)")
    plt.title("Learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/learning_curve.png", dpi=150)


if __name__ == "__main__":
    main()