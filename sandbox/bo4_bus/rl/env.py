"""
BusPriorityEnv: single-agent Gymnasium env, the RL agent controls junction B1 of the
BO4 3x3 grid (where lines L1 and L2 cross). SUMO-RL is not used (tutor rule).

    obs, info = env.reset(seed=0)                 # draws a training seed (default 0-899)
    obs, info = env.reset(options={"sumo_seed": 900})   # explicit seed (validation)
    obs, r, terminated, truncated, info = env.step(action)

Scenario = the existing grid runner (../Grid_Demo.py, ../gen_grid_routes.py): same
network, demand (675 veh/h per entry), closed-loop bus dwell, headless libsumo.
The other 8 junctions run controller C (../rule_tsp.py, C_tuned parameters).

Episode: warm-up (default 600 s, not learned; B1 runs its fixed-time program), then
`duration` s (default 3600) of decision ticks every `delta` = 5 s -> 720 steps.

Reward (reward=): "vehicle" (D), "person" (E), "regular" (F, weight w) or "regular_g"
(G: F + early-bus penalty w_early + switch cost c_switch); see reward.py.
Observation (obs=): "v1" (default, 24 values; the RL-D / RL-E models) or "v2" (36 values);
see observation.py.

Action mask (agent G): env.action_masks() = action_mask(...) of the current state, the
actions the shield would apply unchanged (no switch before min green or during a
transition, forced switch at max green). sb3-contrib MaskablePPO only samples those, so
the shield below stays as the final safety check but should never override.

Action: index k of the next green phase of B1 (0 = E-W, 1 = N-S). Keeping the current
index keeps the green; another index starts a switch. The agent never sets yellow or
red: B1Signal runs yellow 3 s + all-red 2 s (= exactly one tick), then the new green.

Mini shield (overrides logged in env.shield_log and info["shield_rule"]):
    min_green  a switch before the green has run 10 s -> keep
    max_green  keeping a green that would exceed 60 s by the next tick -> switch
A safety monitor checks every simulated second of B1 and records any violation of
yellow == 3 s, all-red == 2 s, 10 s <= green <= 60 s in env.violations (expected empty).

B1Signal (shield + transitions + monitor) is also used by evaluate.py, so the policy
drives B1 identically in training and in the evaluation runs.
"""
import functools
import os
import shutil
import sys
import tempfile
import time

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BO4 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, BO4)
import Grid_Demo as gd  # noqa: E402
import gen_grid_routes as gen  # noqa: E402
import rule_tsp  # noqa: E402
from build_grid import ALL_RED, MAX_GREEN, MIN_GREEN, NET, STOPS_ADD, YELLOW  # noqa: E402
from c_experiments import C_TUNED  # noqa: E402

import observation as ob  # noqa: E402
import reward as rw  # noqa: E402
from sense import GREEN_PROGRAM_IDX, TLS, Geometry, Sensor  # noqa: E402

TRAIN_SEEDS = (0, 900)           # [lo, hi): training seeds; 900-999 kept for validation
YELLOW_IDX = {g + 1 for g in GREEN_PROGRAM_IDX}
ALL_RED_IDX = {g + 2 for g in GREEN_PROGRAM_IDX}
HOLD = 100000.0                  # s, "infinite" duration of a phase the controller holds
EPISODE_KEYS = ("ep_switches", "ep_shield_overrides", "ep_shield_min_green",
                "ep_shield_max_green", "ep_violations", "ep_teleports")


def shield(proposed, current, time_in_phase, n_greens=2, delta=5,
           min_green=MIN_GREEN, max_green=MAX_GREEN):
    """Mini safety shield, pure. Returns (applied action, rule or None, reason or None)."""
    if not 0 <= proposed < n_greens:
        return current, "invalid", f"action {proposed} not a green phase index"
    if proposed != current and time_in_phase < min_green:
        return current, "min_green", f"min green not reached ({time_in_phase:.0f} s of {min_green} s)"
    if proposed == current and time_in_phase + delta > max_green:
        return ((current + 1) % n_greens, "max_green",
                f"max green reached ({time_in_phase:.0f} s of {max_green} s)")
    return proposed, None, None


def action_mask(current, time_in_phase, in_transition=False, n_greens=2, delta=5,
                min_green=MIN_GREEN, max_green=MAX_GREEN):
    """Valid actions under the shield rules, pure: bool array of length n_greens.
    True exactly for the actions the shield applies unchanged, so a policy that only
    picks True actions is never overridden. During yellow / all-red only "keep the
    target" is valid (cannot occur at a decision tick when delta = yellow + all-red)."""
    mask = np.zeros(n_greens, dtype=bool)
    if in_transition or time_in_phase < min_green:
        mask[current] = True                      # no switch before min green
    elif time_in_phase + delta > max_green:
        mask[(current + 1) % n_greens] = True     # forced switch at max green
    else:
        mask[:] = True
    return mask


class B1Signal:
    """Signal controller of the agent's junction: shield, transitions, safety monitor.
    Per simulated second t (after the SUMO step): on_second(sim, t), then at a decision
    tick apply(sim, t, action). Transitions use explicit setPhase calls (SUMO applies a
    programmed switch one step late): yellow YELLOW s, all-red ALL_RED s, new green,
    held until the agent switches again."""

    def __init__(self, tls=TLS, delta=5):
        self.tls, self.delta = tls, delta
        self.n_greens = len(GREEN_PROGRAM_IDX)
        self.active = False
        self.violations, self.shield_log = [], []

    def ready(self, sim):
        return sim.trafficlight.getPhase(self.tls) in GREEN_PROGRAM_IDX

    def take_over(self, sim, t):
        """Take the junction at a green (ready() is True) and hold it."""
        ph = sim.trafficlight.getPhase(self.tls)
        self.current_green = GREEN_PROGRAM_IDX.index(ph)
        self.green_start = t - sim.trafficlight.getSpentDuration(self.tls)
        sim.trafficlight.setPhaseDuration(self.tls, HOLD)
        self._switch_t0 = self._target = None
        self._mon_phase, self._mon_start = ph, self.green_start
        self.active = True

    def time_in_phase(self, t):
        return t - self.green_start

    def mask(self, t):
        """action_mask at time t (decision ticks): the actions apply() leaves unchanged."""
        return action_mask(self.current_green, self.time_in_phase(t), self._switch_t0 is not None,
                           self.n_greens, self.delta)

    def _set(self, sim, phase):
        sim.trafficlight.setPhase(self.tls, phase)
        sim.trafficlight.setPhaseDuration(self.tls, HOLD)

    def apply(self, sim, t, proposed):
        """Shield the action, start a switch if needed. Returns (applied, rule, reason, switch)."""
        proposed = int(proposed)
        applied, rule, reason = shield(proposed, self.current_green, self.time_in_phase(t),
                                       self.n_greens, self.delta)
        if rule:
            self.shield_log.append({"t": t, "proposed": proposed, "applied": applied,
                                    "rule": rule, "reason": reason})
        switch = applied != self.current_green
        if switch:
            self._set(sim, GREEN_PROGRAM_IDX[self.current_green] + 1)       # yellow
            self._switch_t0, self._target = t, applied
            self._monitor(sim, t)
        return applied, rule, reason, switch

    def on_second(self, sim, t):
        if self._switch_t0 is not None:
            k = t - self._switch_t0
            if k == YELLOW:
                self._set(sim, GREEN_PROGRAM_IDX[self.current_green] + 2)   # all-red
            elif k == YELLOW + ALL_RED:
                self._set(sim, GREEN_PROGRAM_IDX[self._target])
                self.current_green, self.green_start = self._target, t
                self._switch_t0 = self._target = None
        self._monitor(sim, t)

    def _monitor(self, sim, t):
        """Record the phase active during [t, t+1) and check the team timing."""
        ph = sim.trafficlight.getPhase(self.tls)
        if ph != self._mon_phase:
            self._check(self._mon_phase, t - self._mon_start, t, ended=True)
            self._mon_phase, self._mon_start = ph, t
        elif ph in GREEN_PROGRAM_IDX and t - self._mon_start > MAX_GREEN:
            # completed seconds only: a switch decided at this same t (apply) still ends it
            self._check(ph, t - self._mon_start, t, ended=False)

    def _check(self, phase, dur, t, ended):
        bad = None
        if phase in YELLOW_IDX and dur != YELLOW:
            bad = f"yellow {dur:.0f} s != {YELLOW}"
        elif phase in ALL_RED_IDX and dur != ALL_RED:
            bad = f"all-red {dur:.0f} s != {ALL_RED}"
        elif phase in GREEN_PROGRAM_IDX and (dur > MAX_GREEN or (ended and dur < MIN_GREEN)):
            bad = f"green {dur:.0f} s outside [{MIN_GREEN}, {MAX_GREEN}]"
        if bad and not (self.violations and self.violations[-1]["t_start"] == t - dur):
            self.violations.append({"t": t, "t_start": t - dur, "phase": phase, "what": bad})


class BusPriorityEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward="vehicle", reward_scale=0.01, warmup=600.0, duration=3600.0,
                 delta=5, entry_flow=675.0, c_params=None, train_seeds=TRAIN_SEEDS,
                 w=1.0, obs="v1", w_early=0.0, c_switch=0.0):
        assert delta == YELLOW + ALL_RED, "a switch must fill exactly one tick"
        self.reward_fn = rw.REWARDS[reward]
        if reward == "regular":
            self.reward_fn = functools.partial(self.reward_fn, w=w)
        elif reward == "regular_g":
            self.reward_fn = functools.partial(self.reward_fn, w=w, w_early=w_early,
                                               c_switch=c_switch)
        self.reward_name = reward
        self.reward_scale = reward_scale
        self.w, self.w_early, self.c_switch = w, w_early, c_switch
        self.obs_version = obs
        self.warmup, self.duration, self.delta = warmup, duration, delta
        self.entry_flow = entry_flow
        self.c_params = dict(c_params or C_TUNED)
        self.train_seeds = train_seeds
        self.observation_space = gym.spaces.Box(*ob.bounds(obs), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(GREEN_PROGRAM_IDX))

        import sumolib
        self.net = sumolib.net.readNet(NET, withInternal=True)
        self.sensor = Sensor(Geometry(self.net))
        self.workdir = tempfile.mkdtemp(prefix="bo4_rl_")
        self.sim = None
        self.sumo_seed = None
        self.signal = B1Signal(TLS, delta)

    # ------------------------------------------------------------------ SUMO
    def _start(self, seed):
        import libsumo as sim
        if self.sim is not None:
            self.sim.close()
        t_end = self.warmup + self.duration
        routes = os.path.join(self.workdir, f"routes_seed{seed}.rou.xml")
        gen.generate(seed, routes, t_end=t_end, entry_flow=self.entry_flow)
        sim.start([gd.sumolib.checkBinary("sumo"), "-n", NET, "-r", routes, "-a", STOPS_ADD,
                   "--seed", str(seed), "--begin", "0", "--end", str(t_end + 1),
                   "--time-to-teleport", "300", "--no-step-log", "true", "--no-warnings", "true"])
        self.sim = sim
        p = rule_tsp.Params(late_thr=self.c_params["late_thr"], queue_thr=self.c_params["queue_thr"],
                            max_ext=self.c_params["max_ext"])
        self.controllers = rule_tsp.make_controllers("tsp_cond", self.net, p, gd.FixedTimeController)
        del self.controllers[TLS]                    # B1 belongs to the agent
        self.rng = np.random.default_rng([seed, 7])  # dwell RNG, as in Grid_Demo
        self.lam = (gd.MEAN_DWELL - gd.T_FIXED) / (gd.T_BOARD * 180.0)
        self.last_arrival, self.release = {}, {}
        self.teleports = 0

    def _sim_step(self):
        """One simulated second: SUMO step, closed-loop dwell, C on the other junctions,
        B1 transitions and safety monitor."""
        sim = self.sim
        sim.simulationStep()
        t = sim.simulation.getTime()
        for v in sim.simulation.getStopStartingVehiclesIDList():
            if not v.startswith(("L1_", "L2_")):
                continue
            stop_id = sim.vehicle.getStops(v, 1)[0].stoppingPlaceID
            gap = t - self.last_arrival.get(stop_id, t - 180.0)
            self.last_arrival[stop_id] = t
            dwell, _ = gd.dwell_time(self.rng, gap, self.lam)
            self.release[v] = t - 1 + dwell
        for v in [v for v, tr in self.release.items() if t >= tr]:
            sim.vehicle.resume(v)
            del self.release[v]
        for c in self.controllers.values():
            c.act(t, sim)
        self.teleports += sim.simulation.getStartingTeleportNumber()
        if self.signal.active:
            self.signal.on_second(sim, t)
        return t

    # ------------------------------------------------------------------ gym API
    current_green = property(lambda self: self.signal.current_green)
    violations = property(lambda self: self.signal.violations)
    shield_log = property(lambda self: self.signal.shield_log)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "sumo_seed" in options:
            self.sumo_seed = int(options["sumo_seed"])
        else:
            self.sumo_seed = int(self.np_random.integers(*self.train_seeds))
        self.signal = B1Signal(TLS, self.delta)
        self._start(self.sumo_seed)
        self.ep_reward, self.ep_steps, self.ep_tic = 0.0, 0, time.time()
        self.ep_switches = 0

        t = self.sim.simulation.getTime()
        while t < self.warmup or not self.signal.ready(self.sim):
            t = self._sim_step()
        self.signal.take_over(self.sim, t)
        self.t_end = t + self.duration
        self.sensor.reset()
        self.state = self._sense(t)
        return self._observe(), {"sumo_seed": self.sumo_seed, "t": t}

    def _observe(self):
        return ob.observe(self.state, version=self.obs_version)

    def _sense(self, t):
        return self.sensor.sense(self.sim, t, self.signal.current_green,
                                 self.signal.time_in_phase(t), False)

    def action_masks(self):
        """Valid actions now (sb3-contrib MaskablePPO reads this through the VecEnv)."""
        return self.signal.mask(self.sim.simulation.getTime())

    def step(self, action):
        t = self.sim.simulation.getTime()
        proposed = int(action)
        applied, rule, _, switch = self.signal.apply(self.sim, t, proposed)
        self.ep_switches += switch
        for _ in range(self.delta):
            t = self._sim_step()

        prev, self.state = self.state, self._sense(t)
        r_raw, terms = self.reward_fn(prev, self.state)
        r = r_raw * self.reward_scale
        self.ep_reward += r
        self.ep_steps += 1
        truncated = t >= self.t_end
        info = {"t": t, "proposed": proposed, "applied": applied, "shield_rule": rule,
                "switch": switch, "reward_raw": r_raw, **terms}
        if truncated:
            info["episode_stats"] = self.episode_stats()
            # flat copies for VecMonitor(info_keywords=EPISODE_KEYS) -> monitor.csv
            for k in EPISODE_KEYS:
                info[k] = info["episode_stats"][k[3:]]
        return self._observe(), r, False, truncated, info

    def episode_stats(self):
        return {"sumo_seed": self.sumo_seed, "steps": self.ep_steps, "reward": self.ep_reward,
                "switches": self.ep_switches,
                "shield_overrides": len(self.shield_log),
                "shield_min_green": sum(1 for s in self.shield_log if s["rule"] == "min_green"),
                "shield_max_green": sum(1 for s in self.shield_log if s["rule"] == "max_green"),
                "violations": len(self.violations), "teleports": self.teleports,
                "wall_s": round(time.time() - self.ep_tic, 1)}

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        shutil.rmtree(self.workdir, ignore_errors=True)
