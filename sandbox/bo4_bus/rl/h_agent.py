"""
Agent H: residual control on top of rule controller C (../rule_tsp.py, unchanged).

Every junction keeps C: the fixed 75 s program plus C's late-bus priority. H only acts
when a bus is within C's detection range (150 m, not dwelling / not about to stop
before the stop line) while a green runs, at decision ticks every DELTA = 5 s:
    0 follow  C decides as usual until the next tick (C's own extend / recall)
    1 serve   priority for the focus bus whatever its lateness: extend its green so it
              clears, or cut the conflicting green to its minimum (C's request_for)
    2 hold    the opposite: cut the focus bus's green to its minimum, or keep the
              conflicting green running at least until the next tick
Focus bus = the nearest detected bus. After serve / hold, C's decision is skipped until
the next tick; with no bus in range H is C. "Always follow" is exactly C.
Safety: every action respects C's limits (green 10-60 s, extensions <= max_ext s per
cycle, C's queue guard on the losing approaches, C's own yellow / all-red);
h_options() lists the feasible actions, action_mask() masks the others, and a
B1Signal monitor checks the team timing every second.

Observation (OBS_SIZE_H = 43): the junction's v2 observation (36, observation.py) +
    focus bus lateness / H clipped to [-2, 2] (0 if unknown), distance / 150 m,
    occupancy / 60, its approach has green, C would extend, C would recall,
    extension budget left / max_ext.

HEnv: training env, H at B1 and C at the other 8 junctions, same scenario as
BusPriorityEnv. One step = one decision of H; the reward is reward_regular_g2 summed
over the 5 s ticks until the next decision (semi-Markov steps).
HController(..., policy=...) is the evaluation controller (evaluate.py, any junction).
"""
import os
import sys

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import observation as ob  # noqa: E402
import reward as rw  # noqa: E402
from env import TLS, B1Signal, BusPriorityEnv  # noqa: E402  (puts bo4_bus/ on sys.path)
import rule_tsp  # noqa: E402
from rule_tsp import CONFLICT, GREEN_PHASES, N_PHASES, decide, request_for, sense  # noqa: E402
from sense import Geometry, Sensor  # noqa: E402

DELTA = 5
FOLLOW, SERVE, HOLD = 0, 1, 2
N_ACTIONS = 3
OBS_SIZE_H = ob.OBS_SIZE_V2 + 7
LOW_H = np.concatenate([ob.LOW_V2, np.array([-2, 0, 0, 0, 0, 0, 0], dtype=np.float32)])
HIGH_H = np.concatenate([ob.HIGH_V2, np.array([2, 1, 1, 1, 1, 1, 1], dtype=np.float32)])


# ------------------------------------------------------------------ pure
def _queue_ok(v, losing, p):
    return v.queues.get(losing, 0) <= p.queue_thr


def h_options(v, bus, p, delta=DELTA):
    """Feasible actions for focus `bus` at JunctionView `v` (a green running), pure.
    {action: (kind, new_remaining, extension seconds)}; FOLLOW is always feasible."""
    opts = {FOLLOW: ("follow", None, 0.0)}
    if v.phase not in GREEN_PHASES:
        return opts
    kind, new_rem, _, _ = request_for(bus, v, p)          # C's own request, any lateness
    if kind == "extend" and _queue_ok(v, CONFLICT[v.phase], p):
        opts[SERVE] = ("extend", new_rem, new_rem - v.remaining)
    elif kind == "recall" and _queue_ok(v, v.phase, p):
        opts[SERVE] = ("recall", new_rem, 0.0)
    if v.phase == bus.phase:                              # bus has green: cut it
        new_rem = max(0.0, p.min_green - v.spent)
        if new_rem < v.remaining and _queue_ok(v, v.phase, p):
            opts[HOLD] = ("cut", new_rem, 0.0)
    else:                                                 # bus has red: keep it red
        ext = max(0.0, delta + 1 - v.remaining)
        if (ext <= p.max_ext - v.ext_used + 1e-9 and v.spent + v.remaining + ext <= p.max_green
                and _queue_ok(v, CONFLICT[v.phase], p)):
            opts[HOLD] = ("hold_red", v.remaining + ext, ext)
    return opts


def action_mask(opts):
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[list(opts)] = True
    return m


def extras(v, bus, p, green_of_bus, c_decision, headway=180.0):
    late = 0.0 if bus.lateness is None else float(np.clip(bus.lateness / headway, -2, 2))
    budget = max(0.0, p.max_ext - v.ext_used) / p.max_ext if p.max_ext else 0.0
    return np.array([late, min(bus.dist / 150.0, 1.0), min(bus.occupancy / 60.0, 1.0),
                     float(green_of_bus), float(c_decision.action == "extend"),
                     float(c_decision.action == "recall"), min(budget, 1.0)], dtype=np.float32)


# ------------------------------------------------------------------ controller
class HController(rule_tsp.TspController):
    """C at one junction with H's residual decisions. act() mirrors
    rule_tsp.TspController.act (sensing, cycle budget, passages, lateness), then lets
    C decide only in "C" mode. policy(obs, mask) -> action, or None when an env drives
    it (then `waiting` is set at a decision tick and apply() is called by the env)."""

    def __init__(self, tls_id, appr, params, occupancy_cache, net, policy=None, warmup=600.0,
                 delta=DELTA, obs_version="h"):
        super().__init__(tls_id, appr, params, occupancy_cache)
        self.policy, self.warmup, self.delta = policy, warmup, delta
        self.obs_version = obs_version
        self.geometry = Geometry(net, tls_id)
        self.sensor = Sensor(self.geometry)
        self.signal = B1Signal(tls_id, delta)          # timing monitor (+ empty shield_log)
        self.mode = "C"
        self.waiting = False
        self.new_tick = False
        self.state = None
        self.switches = 0
        self.counts = {"follow": 0, "serve": 0, "hold": 0}
        self._v = self._focus = self._opts = self._obs = None
        self._phase_seen = None

    # -- monitoring
    def _monitor(self, sim, t, phase):
        """Timing monitor from the first phase change after the warm-up on (the phase
        running at the end of the warm-up is only partly observed, so it is not checked)."""
        if self._phase_seen is not None and phase != self._phase_seen:
            if phase in GREEN_PHASES:
                self.switches += 1
            if not hasattr(self.signal, "_mon_phase"):
                self.signal._mon_phase, self.signal._mon_start = phase, t
        self._phase_seen = phase
        if hasattr(self.signal, "_mon_phase"):
            self.signal._monitor(sim, t)

    # -- C's decision (rule_tsp.TspController.act, decision part)
    def _c_decide(self, t, sim, v):
        d = decide(v, self.p)
        lat = {b.id: b.lateness for b in v.buses}
        for bus, code, reason in d.refusals:
            st = self._pending.setdefault(bus, {"granted": False, "code": None})
            st["code"] = code
            self._record(t, bus, "refuse", code, reason, lat.get(bus))
        if d.action == "none":
            return
        st = self._pending.setdefault(d.bus, {"granted": False, "code": None})
        st["granted"] = True
        seconds = d.new_remaining - v.remaining
        if d.action == "extend":
            self._ext_used += seconds
        self._set_remaining(sim, v.phase, d.new_remaining)
        self._record(t, d.bus, d.action, d.code, d.reason, lat.get(d.bus), round(seconds, 1))

    def _set_remaining(self, sim, phase, new_rem):
        if new_rem <= 0:
            sim.trafficlight.setPhase(self.tls_id, (phase + 1) % N_PHASES)   # its own yellow
        else:
            sim.trafficlight.setPhaseDuration(self.tls_id, new_rem)

    def act(self, t, sim):
        v = sense(sim, self.tls_id, self.appr, t, self.p, self.occ)
        if v.phase == rule_tsp.EW_PHASE and self._last_phase not in (None, rule_tsp.EW_PHASE):
            self._ext_used = 0.0
        self._last_phase = v.phase
        v.ext_used = self._ext_used
        now = {b.id for b in v.buses}
        for bus in self._prev_buses - now:
            self._last_pass[rule_tsp.line_of(bus)] = t
            st = self._pending.pop(bus, None)
            if st is not None:
                self.outcomes.append({"t": t, "tls": self.tls_id, "bus": bus,
                                      "line": rule_tsp.line_of(bus),
                                      "outcome": "granted" if st["granted"] else "refused",
                                      "code": st["code"]})
        self._prev_buses = now
        for b in v.buses:
            last = self._last_pass.get(rule_tsp.line_of(b.id))
            b.lateness = None if last is None else t - last - self.p.headway

        if t < self.warmup:
            self._c_decide(t, sim, v)
            return
        self._monitor(sim, t, v.phase)
        if int(t) % self.delta == 0:                      # decision tick
            green = v.phase in GREEN_PHASES
            self.state = self.sensor.sense(sim, t, GREEN_PHASES.index(v.phase) if green else 0,
                                           v.spent if green else 0.0, not green)
            self.new_tick = True
            self.mode = "C"
            if green and v.buses:
                self._v = v
                self._focus = min(v.buses, key=lambda b: b.dist)
                self._opts = h_options(v, self._focus, self.p, self.delta)
                self._obs = np.concatenate([
                    ob.observe_at(self.state, self.geometry, "v2"),
                    extras(v, self._focus, self.p, v.phase == self._focus.phase,
                           decide(v, self.p), self.p.headway)]).astype(np.float32)
                if self.policy is None:
                    self.waiting = True                   # the env calls apply()
                    return
                self.apply(sim, t, self.policy(self._obs, self.mask()))
                return
        if self.mode == "C":
            self._c_decide(t, sim, v)

    # -- decisions
    def mask(self):
        return action_mask(self._opts) if self._opts else np.ones(N_ACTIONS, dtype=bool)

    def observe(self):
        return self._obs

    def apply(self, sim, t, action):
        action = int(action)
        self.waiting = False
        if action not in self._opts:                      # cannot happen with the mask
            self.signal.shield_log.append({"t": t, "proposed": action, "applied": FOLLOW,
                                           "rule": "h_infeasible", "reason": "not feasible"})
            action = FOLLOW
        kind, new_rem, ext = self._opts[action]
        if action == FOLLOW:
            self.counts["follow"] += 1
            self.mode = "C"
            self._c_decide(t, sim, self._v)
            return action
        self.counts["serve" if action == SERVE else "hold"] += 1
        self.mode = "H"
        self._ext_used += ext
        self._set_remaining(sim, self._v.phase, new_rem)
        self._record(t, self._focus.id, f"h_{kind}", f"h_{kind}", f"H {kind} for {self._focus.id}",
                     self._focus.lateness, round(new_rem - self._v.remaining, 1))
        return action


# ------------------------------------------------------------------ training env
class HEnv(BusPriorityEnv):
    """H at B1 (driven by the agent), C at the other 8 junctions. Reward: regular_g2."""

    def __init__(self, w_delay=1.0, w_headway=4.0, c_switch=0.5, **kw):
        super().__init__(reward="regular_g2", obs="v2", w_delay=w_delay, w_headway=w_headway,
                         c_switch=c_switch, **kw)
        self.observation_space = gym.spaces.Box(LOW_H, HIGH_H, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self._appr = rule_tsp.junction_geometry(self.net)[TLS]

    def _start(self, seed):
        super()._start(seed)                              # C everywhere but B1
        p = rule_tsp.Params(mode="cond", late_thr=self.c_params["late_thr"],
                            queue_thr=self.c_params["queue_thr"], max_ext=self.c_params["max_ext"])
        self.ctrl = HController(TLS, self._appr, p, {}, self.net, policy=None, warmup=self.warmup)
        self.controllers[TLS] = self.ctrl

    def _advance(self):
        """Simulate until H must decide or the episode ends; sum the tick rewards."""
        total, terms_sum = 0.0, {}
        while True:
            t = self._sim_step()
            if self.ctrl.new_tick:
                self.ctrl.new_tick = False
                prev, self.state = self.state, self.ctrl.state
                if prev is not None:
                    r, terms = self.reward_fn(prev, self.state, events=self.ctrl.sensor.passages)
                    total += r
                    for k, x in terms.items():
                        terms_sum[k] = terms_sum.get(k, 0.0) + x
            if self.ctrl.waiting or t >= self.t_end:
                return t, total, terms_sum

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        self.sumo_seed = (int(options["sumo_seed"]) if "sumo_seed" in options
                          else int(self.np_random.integers(*self.train_seeds)))
        self.signal = B1Signal(TLS, self.delta)           # inactive: H / C drive B1
        self._start(self.sumo_seed)
        self.t_end = self.warmup + self.duration
        self.state = None
        self.ep_reward, self.ep_steps, self.ep_switches = 0.0, 0, 0
        import time
        self.ep_tic = time.time()
        t, _, _ = self._advance()
        self._last_obs = self.ctrl.observe() if self.ctrl.waiting else np.zeros(OBS_SIZE_H, np.float32)
        return self._last_obs, {"sumo_seed": self.sumo_seed, "t": t}

    def action_masks(self):
        return self.ctrl.mask() if self.ctrl.waiting else np.ones(N_ACTIONS, dtype=bool)

    def step(self, action):
        t = self.sim.simulation.getTime()
        applied = self.ctrl.apply(self.sim, t, action) if self.ctrl.waiting else FOLLOW
        t, r_raw, terms = self._advance()
        r = r_raw * self.reward_scale
        self.ep_reward += r
        self.ep_steps += 1
        truncated = t >= self.t_end
        if self.ctrl.waiting:
            self._last_obs = self.ctrl.observe()
        info = {"t": t, "proposed": int(action), "applied": applied, "shield_rule": None,
                "switch": False, "reward_raw": r_raw, **terms}
        if truncated:
            info["episode_stats"] = self.episode_stats()
            for k in ("ep_switches", "ep_shield_overrides", "ep_shield_min_green",
                      "ep_shield_max_green", "ep_violations", "ep_teleports"):
                info[k] = info["episode_stats"][k[3:]]
        return self._last_obs, r, False, truncated, info

    def episode_stats(self):
        st = super().episode_stats()
        st.update(switches=self.ctrl.switches, violations=len(self.ctrl.signal.violations),
                  shield_overrides=len(self.ctrl.signal.shield_log), **self.ctrl.counts)
        return st
