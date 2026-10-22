"""
Smoke test of BusPriorityEnv (no training): gymnasium's env checker, then
episodes alternating random actions and "always keep the current phase"
(default 4: random, random, keep, keep).

    python bo4_bus/rl/smoke.py [--reward vehicle|person|regular] [--w 1] [--obs v1|v2]
                               [--duration 3600] [--episodes 4]

Checks per episode: no crash, no safety violation, every observation inside the
observation space, finite rewards. Prints shield overrides, reward statistics
(raw = before reward_scale) and wall-clock time per episode; for reward F also the
share of the lateness term (r_late) in the raw reward.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import BusPriorityEnv  # noqa: E402

POLICIES = ["random", "random", "keep", "keep"]


def run_episode(env, policy, seed):
    obs, info = env.reset(seed=seed)
    assert env.observation_space.contains(obs), obs
    raws, late, out_of_space, switches = [], [], 0, 0
    lo, hi = obs.copy(), obs.copy()
    while True:
        a = env.action_space.sample() if policy == "random" else env.current_green
        obs, r, term, trunc, info = env.step(a)
        out_of_space += not env.observation_space.contains(obs)
        lo, hi = np.minimum(lo, obs), np.maximum(hi, obs)
        assert np.isfinite(r)
        raws.append(info["reward_raw"])
        late.append(info.get("r_late", 0.0))
        switches += info["switch"]
        if term or trunc:
            break
    st = info["episode_stats"]
    raws, late = np.array(raws), np.array(late)
    st.update(policy=policy, obs=env.obs_version, switches=switches,
              obs_out_of_space=out_of_space, obs_min=float(lo.min()), obs_max=float(hi.max()),
              raw_mean=raws.mean(), raw_std=raws.std(), raw_min=raws.min(), raw_max=raws.max())
    if env.reward_name == "regular":
        st.update(late_mean=late.mean(), late_min=late.min(),
                  late_share=late.sum() / raws.sum() if raws.sum() else 0.0)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="vehicle", choices=["vehicle", "person", "regular"])
    ap.add_argument("--w", type=float, default=1.0, help="lateness weight of reward F")
    ap.add_argument("--obs", default="v1", choices=["v1", "v2"])
    ap.add_argument("--duration", type=float, default=3600.0)
    ap.add_argument("--episodes", type=int, default=len(POLICIES))
    a = ap.parse_args()

    env = BusPriorityEnv(reward=a.reward, duration=a.duration, w=a.w, obs=a.obs)
    from gymnasium.utils.env_checker import check_env
    check_env(env, skip_render_check=True)
    print("gymnasium check_env: OK", flush=True)

    rows = []
    for k in range(a.episodes):
        st = run_episode(env, POLICIES[k % len(POLICIES)], seed=k)
        rows.append(st)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}, flush=True)
        if env.violations:
            print("  VIOLATIONS:", env.violations[:5])
        if env.shield_log:
            print("  first shield overrides:", env.shield_log[:2])
    env.close()
    bad = [r for r in rows if r["violations"] or r["obs_out_of_space"]]
    print("\nSMOKE", "FAILED" if bad else "OK",
          f"- mean wall time per episode {np.mean([r['wall_s'] for r in rows]):.1f} s, "
          f"{rows[0]['steps']} steps per episode")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
