"""
Smoke test of BusPriorityEnv (no training): gymnasium's env checker, then
2 episodes with random actions and 2 with "always keep the current phase".

    python bo4_bus/rl/smoke.py [--reward vehicle|person] [--duration 3600]

Checks per episode: no crash, no safety violation, every observation inside the
observation space, finite rewards. Prints shield overrides, reward statistics
(raw = before reward_scale) and wall-clock time per episode.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import BusPriorityEnv  # noqa: E402


def run_episode(env, policy, seed):
    obs, info = env.reset(seed=seed)
    assert env.observation_space.contains(obs), obs
    raws, out_of_space, switches = [], 0, 0
    while True:
        a = env.action_space.sample() if policy == "random" else env.current_green
        obs, r, term, trunc, info = env.step(a)
        out_of_space += not env.observation_space.contains(obs)
        assert np.isfinite(r)
        raws.append(info["reward_raw"])
        switches += info["switch"]
        if term or trunc:
            break
    st = info["episode_stats"]
    raws = np.array(raws)
    st.update(policy=policy, switches=switches, obs_out_of_space=out_of_space,
              raw_mean=raws.mean(), raw_std=raws.std(), raw_min=raws.min(), raw_max=raws.max())
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="vehicle", choices=["vehicle", "person"])
    ap.add_argument("--duration", type=float, default=3600.0)
    a = ap.parse_args()

    env = BusPriorityEnv(reward=a.reward, duration=a.duration)
    from gymnasium.utils.env_checker import check_env
    check_env(env, skip_render_check=True)
    print("gymnasium check_env: OK", flush=True)

    rows = []
    for k, policy in enumerate(["random", "random", "keep", "keep"]):
        st = run_episode(env, policy, seed=k)
        rows.append(st)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items()}, flush=True)
        if env.violations:
            print("  VIOLATIONS:", env.violations[:5])
        if env.shield_log:
            print("  first shield overrides:", env.shield_log[:2])
    env.close()
    bad = [r for r in rows if r["violations"] or r["obs_out_of_space"]]
    print("\nSMOKE", "FAILED" if bad else "OK",
          f"- mean wall time per episode {np.mean([r['wall_s'] for r in rows]):.1f} s, "
          f"{rows[0]['steps']} steps per episode")


if __name__ == "__main__":
    main()
