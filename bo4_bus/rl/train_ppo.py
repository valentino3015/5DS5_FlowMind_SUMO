"""
PPO (stable-baselines3) on BusPriorityEnv. Phase 1 only checks that training runs:

    python bo4_bus/rl/train_ppo.py --steps 5000                   # smoke run
    python bo4_bus/rl/train_ppo.py --steps 200000 --reward person # a real run (later)

Writes bo4_bus/rl/runs/<name>/: progress.csv (SB3 logger: losses per update,
episode reward), monitor.csv (one row per episode), model.zip. Training draws SUMO
seeds 0-999 only (env.reset); evaluation seeds 10000+ are never used here.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env import BusPriorityEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--reward", default="vehicle", choices=["vehicle", "person"])
    ap.add_argument("--n-steps", type=int, default=1024, help="PPO rollout length")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed of PPO and of the training-seed draw")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor

    name = a.name or f"ppo_{a.reward}_{a.steps}"
    out = os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    env = Monitor(BusPriorityEnv(reward=a.reward), os.path.join(out, "monitor.csv"),
                  info_keywords=("episode_stats",))
    model = PPO("MlpPolicy", env, n_steps=a.n_steps, batch_size=64, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, seed=a.seed, verbose=0, device="cpu")
    model.set_logger(configure(out, ["stdout", "csv"]))
    tic = time.time()
    model.learn(total_timesteps=a.steps)
    wall = time.time() - tic
    model.save(os.path.join(out, "model.zip"))
    env.close()
    print(f"trained {model.num_timesteps} steps in {wall:.0f} s "
          f"({wall / model.num_timesteps * 1000:.1f} ms per step); saved {out}")


if __name__ == "__main__":
    main()
