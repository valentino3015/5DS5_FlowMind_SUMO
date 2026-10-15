"""
BusPriorityEnv: single-agent Gymnasium env, the RL agent controls junction B1 of the
BO4 3x3 grid (where lines L1 and L2 cross). SUMO-RL is not used (tutor rule).

    obs, info = env.reset(seed=0)                 # draws a training seed in 0-999
    obs, info = env.reset(options={"sumo_seed": 10000})   # explicit seed (evaluation)
    obs, r, terminated, truncated, info = env.step(action)

Scenario = the existing grid runner (../Grid_Demo.py, ../gen_grid_routes.py): same
network, demand (675 veh/h per entry), closed-loop bus dwell, headless libsumo.
The other 8 junctions run controller C (../rule_tsp.py, C_tuned parameters).

Episode: warm-up (default 600 s, not learned; B1 runs its fixed-time program), then
`duration` s (default 3600) of decision ticks every `delta` = 5 s -> 720 steps.

Action: index k of the next green phase of B1 (0 = E-W, 1 = N-S). Keeping the current
index keeps the green; another index starts a switch. The agent never sets yellow or
red: the env's signal controller runs yellow 3 s + all-red 2 s (= exactly one tick),
then the new green.

Mini shield (overrides logged in env.shield_log and info["shield_rule"]):
    min_green  a switch before the green has run 10 s -> keep
    max_green  keeping a green that would exceed 60 s by the next tick -> switch
A safety monitor checks every simulated second of B1 and records any violation of
yellow == 3 s, all-red == 2 s, 10 s <= green <= 60 s in env.violations (expected empty).
"""
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

TRAIN_SEEDS = (0, 1000)          # [lo, hi): training seeds only
YELLOW_IDX = {g + 1 for g in GREEN_PROGRAM_IDX}
ALL_RED_IDX = {g + 2 for g in GREEN_PROGRAM_IDX}
HOLD = 100000.0                  # s, "infinite" green duration while the env holds a green


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


class BusPriorityEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward="vehicle", reward_scale=0.01, warmup=600.0, duration=3600.0,
                 delta=5, entry_flow=675.0, c_params=None, train_seeds=TRAIN_SEEDS):
        assert delta == YELLOW + ALL_RED, "a switch must fill exactly one tick"
        self.reward_fn = rw.REWARDS[reward]
        self.reward_name = reward
        self.reward_scale = reward_scale
        self.warmup, self.duration, self.delta = warmup, duration, delta
        self.entry_flow = entry_flow
        self.c_params = dict(c_params or C_TUNED)
        self.train_seeds = train_seeds
        self.observation_space = gym.spaces.Box(ob.LOW, ob.HIGH, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(GREEN_PROGRAM_IDX))

        import sumolib
        self.net = sumolib.net.readNet(NET, withInternal=True)
        self.sensor = Sensor(Geometry(self.net))
        self.workdir = tempfile.mkdtemp(prefix="bo4_rl_")
        self.sim = None
        self.sumo_seed = None

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
        B1 safety monitor."""
        sim = self.sim
        if self._monitor_on:      # phase active during [t, t+1): read before the step
            self._monitor(sim.simulation.getTime())
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
        return t

    def _monitor(self, t):
        ph = self.sim.trafficlight.getPhase(TLS)
        if ph != self._mon_phase:
            dur = t - self._mon_start
            self._check(self._mon_phase, dur, t, ended=True)
            self._mon_phase, self._mon_start = ph, t
        elif ph in GREEN_PROGRAM_IDX and t + 1 - self._mon_start > MAX_GREEN:
            self._check(ph, t + 1 - self._mon_start, t, ended=False)

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

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "sumo_seed" in options:
            self.sumo_seed = int(options["sumo_seed"])
        else:
            self.sumo_seed = int(self.np_random.integers(*self.train_seeds))
        self._start(self.sumo_seed)
        self._monitor_on = False
        self.violations, self.shield_log = [], []
        self.ep_reward, self.ep_steps, self.ep_tic = 0.0, 0, time.time()

        t = self.sim.simulation.getTime()
        while t < self.warmup or self.sim.trafficlight.getPhase(TLS) not in GREEN_PROGRAM_IDX:
            t = self._sim_step()
        # take over B1 at a green: hold it until the agent switches
        ph = self.sim.trafficlight.getPhase(TLS)
        self.current_green = GREEN_PROGRAM_IDX.index(ph)
        self.green_start = t - self.sim.trafficlight.getSpentDuration(TLS)
        self.sim.trafficlight.setPhaseDuration(TLS, HOLD)
        self.t_end = t + self.duration
        self._monitor_on = True
        self._mon_phase, self._mon_start = ph, self.green_start
        self.sensor.reset()
        self.state = self._sense(t)
        return ob.observe(self.state), {"sumo_seed": self.sumo_seed, "t": t}

    def _sense(self, t):
        return self.sensor.sense(self.sim, t, self.current_green, t - self.green_start, False)

    def step(self, action):
        sim = self.sim
        t = sim.simulation.getTime()
        proposed = int(action)
        tip = t - self.green_start
        applied, rule, reason = shield(proposed, self.current_green, tip,
                                       len(GREEN_PROGRAM_IDX), self.delta)
        if rule:
            self.shield_log.append({"t": t, "proposed": proposed, "applied": applied,
                                    "rule": rule, "reason": reason})
        switch = applied != self.current_green
        # The env drives every transition with explicit setPhase calls (SUMO applies a
        # programmed switch one step late): yellow for YELLOW s, all-red for ALL_RED s,
        # then the new green, held until the agent switches again.
        yellow = GREEN_PROGRAM_IDX[self.current_green] + 1
        if switch:
            sim.trafficlight.setPhase(TLS, yellow)
            sim.trafficlight.setPhaseDuration(TLS, HOLD)
        for i in range(self.delta):
            if switch and i == YELLOW:
                sim.trafficlight.setPhase(TLS, yellow + 1)
                sim.trafficlight.setPhaseDuration(TLS, HOLD)
            t = self._sim_step()
        if switch:
            sim.trafficlight.setPhase(TLS, GREEN_PROGRAM_IDX[applied])
            sim.trafficlight.setPhaseDuration(TLS, HOLD)
            self.current_green, self.green_start = applied, t

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
        return ob.observe(self.state), r, False, truncated, info

    def episode_stats(self):
        return {"sumo_seed": self.sumo_seed, "steps": self.ep_steps, "reward": self.ep_reward,
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
