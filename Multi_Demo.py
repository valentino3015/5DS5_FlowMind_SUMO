"""
Multi-intersection experiments with SUMO-RL: fixed-time baseline vs independent Q-learning agents.

Examples (run with your venv active):
    python multi_demo.py --list
    python multi_demo.py --scenario cologne3 --episodes 5 --seconds 1800
    python multi_demo.py --scenario grid4x4 --episodes 5 --seconds 1800
    python multi_demo.py --scenario grid4x4 --route 4x4c1.rou.xml
    python multi_demo.py --scenario grid4x4 --rewards diff-waiting-time queue pressure
    python multi_demo.py --scenario cologne3 --gui --episodes 2 --seconds 900

    rewards : diff-waiting-time, queue, average-speed, pressure
"""
import argparse
import os

# Auto-set SUMO_HOME before sumo_rl DO NOT TOUCH OR MODIFY ANYTHING - ONY ADD is imported
if "SUMO_HOME" not in os.environ:
    try:
        import sumo
        os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    except ImportError:
        for _p in (r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Program Files\Eclipse\Sumo"):
            if os.path.isdir(_p):
                os.environ["SUMO_HOME"] = _p
                break

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import xml.etree.ElementTree as ET
import sumo_rl
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

NETS = os.path.join(os.path.dirname(sumo_rl.__file__), "nets")

# scenario -> (folder inside nets/, net file, default route file)
PRESETS = {
    "cologne3": ("RESCO/cologne3", "cologne3.net.xml", "cologne3.rou.xml"),          # 3 real intersections
    "grid4x4": ("4x4-Lucas", "4x4.net.xml", "4x4c1c2c1c2.rou.xml"),                  # 16 intersections
    "arterial4x4": ("RESCO/arterial4x4", "arterial4x4.net.xml", "arterial4x4_1.rou.xml"),
    "single": ("single-intersection", "single-intersection.net.xml", "single-intersection.rou.xml"),
}


def get_first_depart(route_file: str) -> int:
    """Find the earliest departure time in the route file to synchronize the simulation start."""
    try:
        for _, elem in ET.iterparse(route_file, events=("start",)):
            if elem.tag in ("vehicle", "trip"):
                depart = float(elem.attrib.get("depart", 0))
                return int(depart)
    except Exception:
        pass
    return 0


def list_nets():
    print(f"Networks folder: {NETS}\n")
    for root, _, files in os.walk(NETS):
        xml = sorted(f for f in files if f.endswith(".xml"))
        if xml:
            print(os.path.relpath(root, NETS))
            for f in xml:
                print("   ", f)


def make_env(net, route, gui, seconds, fixed, reward, begin_time=0):
    return SumoEnvironment(
        net_file=net,
        route_file=route,
        use_gui=gui,
        begin_time=begin_time,
        num_seconds=seconds,
        min_green=10,
        max_green=50,
        delta_time=5,
        fixed_ts=fixed,
        reward_fn=reward,
        single_agent=False,
        sumo_warnings=False,
        add_per_agent_info=False,
        additional_sumo_cmd="--no-step-log",
    )


def reset(env):
    res = env.reset()
    return res[0] if isinstance(res, tuple) else res


def record(info, begin_time=0):
    return {
        "step": info["step"] - begin_time,
        "waiting_time": info["system_mean_waiting_time"],
        "mean_speed": info["system_mean_speed"],
        "stopped": info["system_total_stopped"],
    }


def run_fixed(net, route, gui, seconds, begin_time=0):
    env = make_env(net, route, gui, seconds, fixed=True, reward="diff-waiting-time", begin_time=begin_time)
    reset(env)
    rows, done = [], {"__all__": False}
    while not done["__all__"]:
        _, _, done, info = env.step({})
        rows.append(record(info, begin_time=begin_time))
    env.close()
    return pd.DataFrame(rows)


def run_ql(net, route, gui, seconds, episodes, reward, begin_time=0, alpha=0.1, gamma=0.99):
    env = make_env(net, route, gui, seconds, fixed=False, reward=reward, begin_time=begin_time)
    agents, runs = None, []
    for ep in range(1, episodes + 1):
        states = reset(env)
        if agents is None:
            agents = {
                ts: QLAgent(
                    starting_state=env.encode(states[ts], ts),
                    state_space=env.observation_space,
                    action_space=env.action_space,
                    alpha=alpha,
                    gamma=gamma,
                    exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=1.0),
                )
                for ts in env.ts_ids
            }
            print(f"   {len(agents)} agents (one per intersection)")
        else:
            for ts in env.ts_ids:
                agents[ts].state = env.encode(states[ts], ts)
                agents[ts].action = None
        rows, done = [], {"__all__": False}
        while not done["__all__"]:
            actions = {ts: agents[ts].act() for ts in agents}
            s, r, done, info = env.step(action=actions)
            for ts in s:
                agents[ts].learn(next_state=env.encode(s[ts], ts), reward=r[ts])
            rows.append(record(info, begin_time=begin_time))
        df = pd.DataFrame(rows)
        runs.append(df)
        print(f"   [{reward}] episode {ep}/{episodes}  mean waiting = {df['waiting_time'].mean():.2f}s  "
              f"mean speed = {df['mean_speed'].mean():.2f} m/s")
    env.close()
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="list available networks and exit")
    p.add_argument("--scenario", default="cologne3", choices=list(PRESETS))
    p.add_argument("--route", default=None, help="route file name inside the scenario folder")
    p.add_argument("--rewards", nargs="+", default=["diff-waiting-time"],
                   help="reward functions: diff-waiting-time queue average-speed pressure")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seconds", type=int, default=1800)
    p.add_argument("--gui", action="store_true")
    args = p.parse_args()

    if args.list:
        list_nets()
        return

    folder, net_name, route_name = PRESETS[args.scenario]
    route_name = args.route or route_name
    net = os.path.join(NETS, folder, net_name)
    route = os.path.join(NETS, folder, route_name)
    for f in (net, route):
        if not os.path.exists(f):
            print(f"File not found: {f}\nRun: python multi_demo.py --list   to see what is available")
            return

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("outputs", exist_ok=True)
    tag = f"{args.scenario}_{os.path.splitext(os.path.splitext(route_name)[0])[0]}"

    begin_time = get_first_depart(route)
    print(f"== Scenario: {args.scenario} | routes: {route_name} | start: 0s (offset: +{begin_time}s) ==")
    print("== Fixed-time baseline ==")
    fixed = run_fixed(net, route, args.gui, args.seconds, begin_time=begin_time)
    fixed.to_csv(f"outputs/{tag}_fixed.csv", index=False)

    results = {"Fixed-time": fixed}
    curves = {}
    for reward in args.rewards:
        print(f"== Q-learning, reward = {reward} ==")
        runs = run_ql(net, route, args.gui, args.seconds, args.episodes, reward, begin_time=begin_time)
        results[f"QL ({reward})"] = runs[-1]
        curves[reward] = [r["waiting_time"].mean() for r in runs]
        runs[-1].to_csv(f"outputs/{tag}_ql_{reward}.csv", index=False)

    # Summary table
    summary = pd.DataFrame(
        {
            name: {
                "mean waiting time (s)": df["waiting_time"].mean(),
                "mean speed (m/s)": df["mean_speed"].mean(),
                "avg stopped vehicles": df["stopped"].mean(),
            }
            for name, df in results.items()
        }
    ).T.round(2)
    base = summary.loc["Fixed-time", "mean waiting time (s)"]
    summary["waiting time vs baseline (%)"] = (100 * (summary["mean waiting time (s)"] - base) / base).round(1)
    summary.to_csv(f"outputs/{tag}_summary.csv")
    print("\n" + summary.to_string())

    # Plots
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for name, df in results.items():
        ax[0].plot(df["step"], df["waiting_time"].rolling(5, min_periods=1).mean(), label=name)
        ax[1].plot(df["step"], df["mean_speed"].rolling(5, min_periods=1).mean(), label=name)
        ax[2].plot(df["step"], df["stopped"].rolling(5, min_periods=1).mean(), label=name)
    for a, t, y in zip(ax, ["Mean waiting time", "Mean speed", "Stopped vehicles"],
                       ["seconds", "m/s", "vehicles"]):
        a.set_title(t)
        a.set_xlabel("Simulation step (s)")
        a.set_ylabel(y)
    ax[0].legend()
    fig.suptitle(f"{args.scenario} - {route_name}")
    plt.tight_layout()
    plt.savefig(f"outputs/{tag}_comparison.png", dpi=150)

    plt.figure(figsize=(6.5, 4))
    for reward, c in curves.items():
        plt.plot(range(1, len(c) + 1), c, marker="o", label=reward)
    plt.axhline(base, color="red", linestyle="--", label="Fixed-time baseline")
    plt.xlabel("Episode")
    plt.ylabel("Mean waiting time (s)")
    plt.title("Learning curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{tag}_learning_curve.png", dpi=150)

    print(f"\nSaved plots, CSVs and summary in outputs/ (prefix: {tag})")


if __name__ == "__main__":
    main()