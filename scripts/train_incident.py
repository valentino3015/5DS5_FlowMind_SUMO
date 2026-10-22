"""Train the PPO response of the incident agent.

Usage:
    python scripts/train_incident.py                 (settings from config.yaml, incident.train)
    python scripts/train_incident.py --timesteps 2000 --n-envs 2   (quick check)
    python scripts/train_incident.py --resume --timesteps 30000    (continue the saved model)

Notion: "Anomaly detector + RL (PPO) for the response". The detector stays
rule-based (IncidentAgent.wants_control). PPO only learns act(): which green
phase to serve while an incident is active.

One episode:
    1. pick a random incoming road (long enough to hold a queue) and block it:
       one broken-down car per lane, written to runs/incident_train/ (random
       start and duration). The whole road, because on a multi-lane road a
       single stopped car is just overtaken and no jam forms.
    2. run cologne3 with fixed-time lights; the rule-based detector watches
    3. when the detector claims a junction, PPO decides that junction's phase
    4. reward = minus the vehicles waiting on lanes that can move,
       minus a penalty if PPO picks a phase it may not use (the rule is used then)

Training seeds start at incident.train.seed; evaluation uses other seeds.

Owner: Donia.
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import os
import sys

import gymnasium as gym
import numpy as np
import sumolib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmind.agents.baselines import FixedTime
from flowmind.agents.incident import IncidentAgent
from flowmind.env import SumoCorridorEnv
from scripts.demo_incident import SCENARIO, load_config, read_scenario

WORK_DIR = ROOT / "runs" / "incident_train"


def incident_roads(net_file: Path, min_length: float) -> list[tuple[str, int, float]]:
    """Incoming roads of the traffic lights, long enough to hold a queue: [(edge_id, lanes, length)]."""
    net = sumolib.net.readNet(str(net_file))
    roads = {}
    for tls in net.getTrafficLights():
        for conn in tls.getConnections():
            edge = conn[0].getEdge()
            if edge.getLength() >= min_length:
                roads[edge.getID()] = (edge.getLaneNumber(), edge.getLength())
    return [(e, n, length) for e, (n, length) in sorted(roads.items())]


def write_block_file(path: Path, edge: str, lanes: int, length: float, start: float, end: float) -> None:
    """One car per lane breaks down 5 m before the stop line of edge, from start until end."""
    cars = "".join(
        f'    <vehicle id="incident_car_{i}" type="broken_down" depart="{start:.0f}" departLane="{i}">\n'
        f'        <route edges="{edge}"/>\n'
        f'        <stop lane="{edge}_{i}" endPos="{length - 5:.1f}" until="{end:.0f}"/>\n'
        '    </vehicle>\n'
        for i in range(lanes)
    )
    path.write_text(
        '<routes>\n'
        '    <vType id="broken_down" vClass="passenger" length="4.3" minGap="1.5" color="1,0,0"/>\n'
        + cars + '</routes>\n',
        encoding="utf-8",
    )


class IncidentTrainEnv(gym.Env):
    """Gym view of the corridor: one step = one PPO decision at a junction in incident."""

    def __init__(self, config: dict, rank: int = 0):
        super().__init__()
        self.params = dict(config["incident"], use_ppo=False)  # detector + rule fallback
        self.train = self.params["train"]
        self.timers_cfg = config["fixed_time"]
        sc = read_scenario(SCENARIO)
        base_routes = [r for r in sc["routes"] if not r.name.endswith(".block.xml")]
        self.begin = sc["begin"]
        self.roads = incident_roads(sc["net"], self.train["min_road_length"])

        WORK_DIR.mkdir(parents=True, exist_ok=True)
        self.block_file = WORK_DIR / f"block_{rank}_{os.getpid()}.xml"
        routes = ",".join(str(r) for r in base_routes + [self.block_file])
        self.env = SumoCorridorEnv(config=config, net_file=sc["net"], route_file=routes,
                                   use_gui=False, begin_time=self.begin)

        n = int(self.params["max_phases"])
        self.observation_space = gym.spaces.Box(0.0, 5.0, shape=(6 * n + 1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n)
        self.rng = np.random.default_rng(int(self.train["seed"]) + 1000 * rank)
        self.no_claim_episodes = 0

    # --- episode -------------------------------------------------------
    def _new_episode(self) -> None:
        edge, lanes, length = self.roads[self.rng.integers(len(self.roads))]
        start = self.begin + self.rng.uniform(*self.train["block_start"])
        end = start + self.rng.uniform(*self.train["block_duration"])
        write_block_file(self.block_file, edge, lanes, length, start, end)
        self.env.seed = int(self.rng.integers(self.train["seed"], self.train["seed"] + 10**6))
        self.env.sim_seconds = int(end - self.begin + self.train["after_block"])
        self.states = self.env.reset()
        self.det = {t: IncidentAgent(self.params, self.env.get_phase_movements(t))
                    for t in self.env.tls_ids}
        self.timers = {t: FixedTime(self.timers_cfg) for t in self.env.tls_ids}
        self.rl_tls = None
        self._observe()

    def _observe(self) -> bool:
        """Run every detector once for this tick; PPO drives the first junction that claims."""
        self.claims = {t: self.det[t].wants_control(st, []) for t, st in self.states.items()}
        claimed = [t for t, c in self.claims.items() if c]
        if claimed:
            self.rl_tls = claimed[0]
        return bool(claimed)

    def _actions(self, rl_action: int | None) -> tuple[dict, bool]:
        actions, unsafe = {}, False
        for tls, st in self.states.items():
            allowed = [True] * st.num_phases
            det = self.det[tls]
            if tls == self.rl_tls and rl_action is not None:
                safe = det.safe_phases(st, allowed)
                if rl_action < len(safe) and safe[rl_action]:
                    actions[tls] = rl_action
                else:
                    actions[tls], unsafe = det.rule_act(st, [], allowed), True
            elif self.claims[tls]:
                actions[tls] = det.rule_act(st, [], allowed)
            else:
                actions[tls] = self.timers[tls].act(st, [], allowed)
        return actions, unsafe

    def _run_until_claim(self) -> bool:
        """Advance with the normal controllers until a junction is in incident. False if the episode ended."""
        done = False
        while not any(self.claims.values()):
            if done:
                return False
            actions, _ = self._actions(None)
            self.states, done, _ = self.env.step(actions)
            self._observe()
        return not done

    def _reward(self, tls: str) -> float:
        det, st = self.det[tls], self.states[tls]
        blocked = set(det.blocked)
        waiting = sum(l.queue for l in st.lanes if l.lane_id not in blocked)
        return -waiting / float(self.params["norm_queue"])

    # --- gym API -------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        while True:
            self._new_episode()
            if self._run_until_claim():
                break
            self.no_claim_episodes += 1  # breakdown never detected: try another one
        return self.det[self.rl_tls].encode(self.states[self.rl_tls]), {}

    def step(self, action: int):
        tls = self.rl_tls
        actions, unsafe = self._actions(int(action))
        self.states, done, _ = self.env.step(actions)
        reward = self._reward(tls) - float(self.params["unsafe_penalty"]) * unsafe
        self._observe()
        if not done and not any(self.claims.values()):
            done = not self._run_until_claim()
        tls = self.rl_tls or tls
        obs = self.det[tls].encode(self.states[tls])
        return obs, reward, False, done, {"unsafe": unsafe, "no_claim_episodes": self.no_claim_episodes}

    def close(self):
        self.env.close()


def make_env(config: dict, rank: int) -> IncidentTrainEnv:
    return IncidentTrainEnv(config, rank)


def main():
    parser = argparse.ArgumentParser(description="Train the incident agent's PPO response")
    parser.add_argument("--timesteps", type=int, default=None, help="Override incident.train.timesteps")
    parser.add_argument("--n-envs", type=int, default=None, help="Override incident.train.n_envs")
    parser.add_argument("--resume", action="store_true",
                        help="Continue training the saved model instead of starting from zero")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    config = load_config()
    params, train = config["incident"], config["incident"]["train"]
    timesteps = args.timesteps or int(train["timesteps"])
    n_envs = args.n_envs or int(train["n_envs"])
    model_path = ROOT / params["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("== Incident agent: PPO training ==")
    print(f"Decisions: {timesteps} | Parallel simulations: {n_envs} | Model: {model_path}\n")

    resume = args.resume and model_path.is_file()
    # When resuming, shift the ranks so the new episodes differ from the first run's.
    first_rank = 100 if resume else 0
    venv = VecMonitor(SubprocVecEnv([partial(make_env, config, first_rank + i) for i in range(n_envs)]))
    if resume:
        model = PPO.load(str(model_path), env=venv)
        print(f"Resuming from {model_path} ({model.num_timesteps} decisions already learned)\n")
    else:
        model = PPO(
            "MlpPolicy", venv, verbose=1, seed=int(train["seed"]),
            n_steps=int(train["n_steps"]), batch_size=int(train["batch_size"]),
            learning_rate=float(train["learning_rate"]), gamma=float(train["gamma"]),
            ent_coef=float(train["ent_coef"]),
        )
    model.set_logger(configure(str(WORK_DIR / "log"), ["stdout", "csv"]))
    checkpoints = CheckpointCallback(
        save_freq=max(1, int(train["checkpoint_every"]) // n_envs),
        save_path=str(WORK_DIR / "checkpoints"), name_prefix="incident_ppo",
    )
    try:
        model.learn(total_timesteps=timesteps, callback=checkpoints, reset_num_timesteps=not resume)
    finally:
        model.save(str(model_path))
        venv.close()
    print(f"\nModel saved to {model_path}")
    print("Learning curve: runs/incident_train/log/progress.csv (ep_rew_mean should rise)")


if __name__ == "__main__":
    main()
