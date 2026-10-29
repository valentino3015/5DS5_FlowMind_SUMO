"""Train/fine-tune an emission-only PPO policy on SUMO, with held-out validation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import sys
import uuid
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from flowmind.env import ROOT, SumoCorridorEnv, simulation_protocol
from flowmind.agents.baselines import FixedTime
from flowmind.emission_rl import MODEL_DIR, METRICS, SAMPLE_KEYS, SafetyGate, encode, topology, training_safety, decode_actions
from flowmind.evaluation import SCENARIOS, digest_files, emission_protocol
from flowmind.demands import case_key, check_disjoint, describe, load_demands, validation_cases
from flowmind.kpis import atomic_json, write_csv


class EmissionTrainingEnv(gym.Env):
    def __init__(self, config, manifest, seeds, demands=None, sample=True, demand_seed=0):
        super().__init__()
        self.config, self.manifest, self.seeds = config, manifest, seeds
        # Route files: drawn at random for training, taken in order (aligned
        # with seeds) for validation. None keeps the scenario route.
        self.demands = [Path(d) for d in demands] if demands else None
        self.sample = sample
        self.rng = np.random.default_rng(demand_seed)
        self.episode_demand = None
        self.index = 0
        self.env = None
        self.samples, self.episode_rows = [], []
        self.action_space = gym.spaces.MultiDiscrete(
            [len(manifest["green_durations_s"])] if manifest.get("coordinated_action") else [
            len(manifest["green_durations_s"]) if manifest.get("action_mode") == "duration" else t["phases"]
            for t in manifest["topology"]])
        size = sum(t["phases"] * 5 + 1 for t in manifest["topology"])
        self.observation_space = gym.spaces.Box(0, np.inf, (size,), np.float32)
        self.total_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        episode_seed = self.seeds[self.index % len(self.seeds)]
        route = None
        if self.demands:
            route = (self.demands[int(self.rng.integers(len(self.demands)))] if self.sample
                     else self.demands[self.index % len(self.demands)])
        self.episode_demand = route.name if route else None
        self.index += 1
        self.samples = []
        self.env = SumoCorridorEnv(config=copy.deepcopy(self.config), route_file=route, seed=episode_seed, use_gui=False)
        self.env.metrics_callback = self.samples.append
        self.states = self.env.reset()
        self.episode_stats = {k: sum(s[k] for s in self.samples) for k in ("teleports", "collisions", "departed", "arrived")}
        self.episode_quantities = [0.] * 4
        self.samples.clear()  # Initial second precedes the first decision.
        self.gate = SafetyGate(self.manifest["safety"])
        self.total_reward, self.steps = 0, 0
        self.episode_seed = episode_seed
        return encode(self.states, self.manifest), {}

    def step(self, action):
        proposed = decode_actions(self.states, action, self.manifest)
        safe, _ = self.gate.apply(self.states, proposed)
        self.samples.clear()
        self.states, _, info = self.env.step(safe)
        totals = [sum(s[k] for s in self.samples) for k in SAMPLE_KEYS]
        before = self.episode_quantities
        self.episode_quantities = [a+b for a,b in zip(before, totals)]
        for key in self.episode_stats:
            self.episode_stats[key] += sum(s[key] for s in self.samples)
        scales = self.reward_scales()
        reward = environmental_reward(totals, scales, self.manifest["weights"])
        if self.manifest.get("reward_mode") == "worst_interval":
            # Add costs for individual seconds, including every transition.
            # The observation contains current rates, so this objective avoids
            # the unobserved episode-history dependency of balanced_cost.
            reward = -sum(worst_interval_cost([sample[k] for k in SAMPLE_KEYS],
                          scales) for sample in self.samples)
        if self.manifest.get("reward_mode") == "balanced":
            # A potential difference telescopes to the whole-episode objective.
            # No waiting/queues/vehicle counts enter this emission-only cost.
            args = (scales, self.manifest["weights"], self.manifest["seconds"])
            reward = balanced_cost(before, *args) - balanced_cost(self.episode_quantities, *args)
        if self.manifest.get("reward_mode") == "pm_tolerant":
            reward = -sum(pm_tolerant_cost([sample[k] for k in SAMPLE_KEYS], scales,
                                           self.manifest["pm_tolerance"], self.manifest["pm_penalty"])
                          for sample in self.samples)
        self.total_reward += reward
        self.steps += 1
        truncated = info["time"] - self.env.begin_time >= self.env.sim_seconds
        if truncated:
            self.episode_rows.append({"episode": len(self.episode_rows)+1, "seed": self.episode_seed,
                                      "demand": self.episode_demand,
                                      "reward": self.total_reward, "decisions": self.steps})
        if self.manifest.get("reward_mode") in ("worst_interval", "pm_tolerant"):
            # A finite evaluation window is a terminal task, not a timeout
            # with hypothetical future emissions to bootstrap.
            return encode(self.states, self.manifest), reward, truncated, False, {}
        return encode(self.states, self.manifest), reward, False, truncated, {}

    def reward_scales(self):
        """FixedTime rates of this episode's demand when known, else the global ones.

        Observations keep the global scales: only the reward is made relative
        to the classical signals on the same traffic.
        """
        return self.manifest.get("demand_scales_mg_s", {}).get(self.episode_demand, self.manifest["scales_mg_s"])

    def close(self):
        if self.env:
            self.env.close()
            self.env = None


def environmental_reward(totals_mg, scales_mg_s, weights):
    """Negative weighted emission quantities; fixed training-only normalizers.

    Every second of yellow/red is included. No division by interval length:
    otherwise longer transitions could artificially improve episode reward.
    """
    return -sum(w * quantity / scale for w, quantity, scale in zip(weights, totals_mg, scales_mg_s)) / sum(weights)


def balanced_cost(quantities, scales, weights, seconds):
    """Average plus worst normalized cumulative emission, fixed train scales.

    Equal weight to worst metric and weighted mean; PM retains its mean weight.
    This reduces compensation between pollutants, without promising four wins.
    """
    ratios = np.asarray(quantities, dtype=float) / (np.asarray(scales) * seconds)
    return 100. * (.5 * np.average(ratios, weights=weights) + .5 * np.max(ratios))


def worst_interval_cost(rates, scales):
    """Four equal environmental objectives, with a convex excess penalty.

    Scales are frozen training-only reference rates. A high pollutant cannot
    disappear inside a weighted mean. This remains a surrogate objective;
    only held-out whole-window totals establish whether all four improved.
    """
    ratios = np.asarray(rates, dtype=float) / np.asarray(scales, dtype=float)
    return float(.5 * np.mean(ratios) + .5 * np.max(ratios)
                 + np.mean(np.maximum(ratios - 1., 0.) ** 2))


def fixed_time_samples(config, seed, route=None):
    """Per-second metrics of one FixedTime run, with the junction layout."""
    samples = []
    env = SumoCorridorEnv(config=copy.deepcopy(config), route_file=route, seed=seed, use_gui=False)
    env.metrics_callback = samples.append
    try:
        states = env.reset()
        layout = topology(states)
        agents = {k: FixedTime(config["fixed_time"]) for k in states}
        while samples[-1]["elapsed_s"] < env.sim_seconds:
            states, _, _ = env.step({k:a.act(states[k], [], [True]*states[k].num_phases) for k,a in agents.items()})
    finally:
        env.close()
    return samples, layout


def reference_rates(samples):
    # Fixed reference rates; floors avoid division by zero for absent pollutants.
    return [max(1.0, sum(s[k] for s in samples)/len(samples)) for k in SAMPLE_KEYS]


def pm_tolerant_cost(rates, scales, tolerance, penalty):
    """Mean CO2/NOx/fuel ratio, plus a linear penalty when PM exceeds 1 + tolerance.

    Ratios use the classical (30 s) FixedTime reference rates. The PM term is a
    per-second surrogate; only whole-window validation totals decide whether
    the PM tolerance is respected.
    """
    ratios = np.asarray(rates, dtype=float) / np.asarray(scales, dtype=float)
    return float(np.mean(ratios[:3]) + penalty * max(ratios[3] - 1. - tolerance, 0.))


DEPARTURE_NOISE = 1  # vehicles: SUMO run-to-run noise, not held-back traffic


def tolerant_selection(comparison, tolerance):
    """Feasibility first (PM within tolerance, no traffic loss), then lowest CO2/NOx/fuel."""
    feasible = (comparison["pm_ratio"] <= 1. + tolerance and comparison["arrival_shortfall_ratio"] <= 1.
                and comparison["departure_gap"] <= DEPARTURE_NOISE)
    return feasible, -float(np.mean([comparison[f"{k}_ratio"] for k in ("co2", "nox", "fuel")]))


def calibrate(config, seeds, demands=None, count=10):
    """FixedTime reference rates; with varied demands, on the first training files."""
    samples, layout = [], None
    for seed, route in validation_cases(seeds, (demands or [])[:count]):
        run, layout = fixed_time_samples(config, seed, route)
        samples += run
    return layout, reference_rates(samples)


def _demand_rates(config, seed, route):
    return Path(route).name, reference_rates(fixed_time_samples(config, seed, route)[0])


def demand_scales(config, seed, demands, workers=1):
    """FixedTime rates of every training demand, one run each, training seed only.

    Separate processes each own their default TraCI connection.
    """
    jobs = ([copy.deepcopy(config)] * len(demands), [seed] * len(demands), list(demands))
    if workers <= 1:
        return dict(map(_demand_rates, *jobs))
    with ProcessPoolExecutor(workers) as pool:
        return dict(pool.map(_demand_rates, *jobs))


def init_duration_policy(policy, neutral=False, durations=(20, 25, 30, 35)):
    """Start a new duration policy from zero weights.

    Default: a preference for the classical 30 s (initialization, not
    training). Neutral: uniform over 20/25/30/35 s, so that the most probable
    action, used in validation, can change as soon as learning moves it.
    """
    import torch
    durations = list(durations)
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.zero_()
        if not neutral and 30 in durations:
            policy.action_net.bias.reshape(-1, len(durations))[:, durations.index(30)] = 0.75


def official_emissions(path):
    """Whole-window SUMO totals, including unfinished trips, in mg."""
    trips = ET.parse(path).getroot().findall("tripinfo")
    emissions = [t.find("emissions") for t in trips]
    if any(e is None for e in emissions):
        raise ValueError("Bilan d’émissions absent pendant la validation")
    values = [sum(float(e.get(key, "0")) for e in emissions)
              for key in ("CO2_abs", "NOx_abs", "fuel_abs", "PMx_abs")]
    if not all(np.isfinite(x) and x >= 0 for x in values):
        raise ValueError("Bilan d’émissions invalide pendant la validation")
    return {"emissions_mg": values, "departed": len(trips),
            "arrived": sum(float(t.get("arrival", "-1")) >= 0 for t in trips)}


def validation_baseline(config, seeds, output, demands=None, name="validation_baseline"):
    """Reference used for model selection, never for learning or normalization."""
    result = {}
    for seed, route in validation_cases(seeds, demands):
        key = case_key(seed, route)
        directory = output / name / f"seed_{key}"
        cfg = copy.deepcopy(config)
        cfg["evaluation_output"] = str(directory)
        env = SumoCorridorEnv(config=cfg, route_file=route, seed=seed, use_gui=False)
        samples = []
        env.metrics_callback = samples.append
        try:
            states = env.reset()
            agents = {key: FixedTime(config["fixed_time"]) for key in states}
            while samples[-1]["elapsed_s"] < env.sim_seconds:
                states, _, _ = env.step({key: agent.act(states[key], [], [True]*states[key].num_phases)
                                        for key, agent in agents.items()})
        finally:
            env.close()
        result[key] = official_emissions(directory / "tripinfo.xml")
        print(f"Référence de validation ({config['fixed_time']['green']} s) : {key}, arrivées={result[key]['arrived']}", flush=True)
    atomic_json(output / f"{name}.json", {"green_s": config["fixed_time"]["green"], "seeds": result})
    return result


def compare_validation(measured, baseline):
    """Paired ratios; discourage apparent savings from holding back traffic."""
    ratios = [float(np.mean([row["emissions_mg"][i] / max(1., baseline[seed]["emissions_mg"][i])
                            for seed, row in measured.items()])) for i in range(4)]
    arrivals = float(np.mean([baseline[seed]["arrived"] / max(1, row["arrived"])
                             for seed, row in measured.items()]))
    departures = float(np.mean([baseline[seed]["departed"] / max(1, row["departed"])
                               for seed, row in measured.items()]))
    return {**{f"{name}_ratio": r for name, r in zip(METRICS, ratios)},
            "arrival_shortfall_ratio": arrivals, "departure_shortfall_ratio": departures,
            "departure_gap": sum(baseline[seed]["departed"] - row["departed"] for seed, row in measured.items()),
            "worst_ratio": max(1., *ratios, arrivals, departures),
            "all_emissions_lower": all(r < 1 for r in ratios)}


class Validate(BaseCallback):
    def __init__(self, raw, config, manifest, seeds, output, every, baseline=None, demands=None, target=None):
        super().__init__()
        self.raw, self.config, self.manifest = raw, config, manifest
        self.seeds, self.output, self.every = seeds, output, every
        self.demands = demands
        self.cases = validation_cases(seeds, demands)
        self.best, self.rows = -np.inf, []
        self.best_timesteps = None
        self.best_key = None
        self.best_teleports = self.best_collisions = None
        self.baseline = baseline
        self.target = target  # optional second FixedTime reference to beat
        self.best_comparison = None

    def assess(self):
        # Training and validation share the default TraCI connection: close first.
        self.raw.close()
        validation = EmissionTrainingEnv(self.config, self.manifest, [seed for seed, _ in self.cases],
                                         [route for _, route in self.cases] if self.demands else None, False)
        scores, diagnostics, quantities, measured = [], [], [], {}
        try:
            for seed, route in self.cases:
                key = case_key(seed, route)
                if self.baseline is not None:
                    directory = self.output / "validation_trials" / f"step_{self.num_timesteps}" / f"seed_{key}"
                    validation.config = copy.deepcopy(self.config)
                    validation.config["evaluation_output"] = str(directory)
                obs, _ = validation.reset()
                finished = False
                while not finished:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = validation.step(action)
                    finished = terminated or truncated
                scores.append(validation.total_reward)
                diagnostics.append(getattr(validation, "episode_stats", {}))
                quantities.append(getattr(validation, "episode_quantities", [0.] * 4))
                if self.baseline is not None:
                    validation.env.close()
                    measured[key] = official_emissions(directory / "tripinfo.xml")
        finally:
            validation.close()
        score = float(np.mean(scores))
        totals = {k: sum(d.get(k, 0) for d in diagnostics) for k in ("teleports", "collisions", "departed", "arrived")}
        comparison = compare_validation(measured, self.baseline) if self.baseline is not None else {}
        if self.target is not None and comparison:
            target = compare_validation(measured, self.target)
            comparison.update({f"target_{k}_ratio": target[f"{k}_ratio"] for k in METRICS})
            comparison["beats_target"] = all(target[f"{k}_ratio"] < 1 for k in ("co2", "nox", "fuel"))
        self.rows.append({"timesteps": self.num_timesteps, "validation_reward": score, **totals,
                          **{f"{name}_mg": float(np.mean([q[i] for q in quantities])) for i,name in enumerate(METRICS)},
                          **comparison})
        write_csv(self.output / "validation.csv", self.rows)
        write_csv(self.output / "training_episodes.csv", self.raw.episode_rows)
        selection_key = (-totals["collisions"], -totals["teleports"],
                         -comparison.get("worst_ratio", 1.), score)
        if self.manifest.get("reward_mode") == "worst_interval" and comparison:
            selection_key = selection_key[:3] + (-max(comparison[f"{k}_ratio"] for k in METRICS), score)
        if self.manifest.get("reward_mode") == "pm_tolerant" and comparison:
            selection_key = selection_key[:2] + tolerant_selection(comparison, self.manifest["pm_tolerance"]) + (score,)
        if self.best_key is None or selection_key > self.best_key:
            self.best_key = selection_key
            self.best = score
            self.best_teleports, self.best_collisions = totals["teleports"], totals["collisions"]
            self.best_timesteps = self.num_timesteps
            self.best_comparison = comparison
            self.model.save(self.output / "best.zip")
        print(f"Validation step={self.num_timesteps}, reward={score:.4f}, teleports={totals['teleports']}, collisions={totals['collisions']}, best={self.best:.4f}", flush=True)
        if comparison:
            print(f"Ratios face aux classiques : {comparison}", flush=True)

    def _on_step(self):
        if self.num_timesteps % 64 == 0:
            atomic_json(self.output / "progress.json", {
                "timesteps": self.num_timesteps,
                "completed_episodes": len(self.raw.episode_rows),
                "current_seed": self.raw.episode_seed,
                "current_demand": getattr(self.raw, "episode_demand", None),
                "best_validation_reward": self.best if np.isfinite(self.best) else None,
            })
            print(f"Progression : {self.num_timesteps} pas, {len(self.raw.episode_rows)} épisodes terminés", flush=True)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=SCENARIOS, default="3x3grid")
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument("--timesteps", type=int, default=20000)
    p.add_argument("--validation-every", type=int, default=2048)
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0,1,2])
    p.add_argument("--validation-seeds", nargs="+", type=int, default=[1000,1001])
    p.add_argument("--weights", nargs=4, type=float, metavar=("CO2","NOX","FUEL","PM"), help="Defaults to emission_rl.weights in config.yaml")
    p.add_argument("--resume", type=Path, help="Existing emission model directory, with compatible manifest")
    p.add_argument("--action-mode", choices=["phase", "duration"], default="phase")
    p.add_argument("--reward-mode", choices=["weighted", "balanced", "worst_interval", "pm_tolerant"], default="weighted")
    p.add_argument("--pm-tolerance", type=float, default=.02, help="pm_tolerant: allowed PM increase over classical signals (0.02 = +2 %%)")
    p.add_argument("--pm-penalty", type=float, default=4., help="pm_tolerant: weight of the PM excess per second")
    p.add_argument("--green-durations", nargs="+", type=int, help="duration mode: candidate green durations in s (default 20 25 30 35)")
    p.add_argument("--coordinated", action="store_true", help="duration mode: one green duration for all junctions")
    p.add_argument("--target-green", type=int, help="Second FixedTime validation reference to beat, e.g. 20")
    p.add_argument("--adapt-reward", action="store_true", help="Explicitly change reward while resuming compatible policy weights")
    p.add_argument("--output", type=Path)
    p.add_argument("--train-demands", type=Path, help="Folder of generated training routes (scripts/generate_demands.py)")
    p.add_argument("--validation-demands", type=Path, help="Folder of generated validation routes")
    p.add_argument("--demand-seed", type=int, default=0, help="Seed of the random draw of one training route per episode")
    p.add_argument("--per-demand-reward", action="store_true",
                   help="Normalize the reward by FixedTime rates of each training demand")
    p.add_argument("--workers", type=int, default=4, help="Processes for the per-demand FixedTime references")
    p.add_argument("--neutral-init", action="store_true", help="New duration policy: uniform start instead of preferring 30 s")
    # PPO hyperparameters, new networks only; defaults keep earlier behaviour.
    p.add_argument("--n-steps", type=int, help="Rollout length (default 64)")
    p.add_argument("--batch-size", type=int, help="Minibatch size (default 32)")
    p.add_argument("--n-epochs", type=int, help="Optimisation passes per rollout (default 4)")
    p.add_argument("--learning-rate", type=float, help="Default 3e-4, or 1e-4 with worst_interval")
    p.add_argument("--linear-lr", action="store_true", help="Linear decay of the learning rate to 0 over --timesteps")
    p.add_argument("--gamma", type=float, help="Default 0.99, or 1 with worst_interval")
    p.add_argument("--gae-lambda", type=float, help="Default 0.95")
    p.add_argument("--ent-coef", type=float, help="Entropy bonus (default 0)")
    args = p.parse_args()
    ppo_args = {k: getattr(args, k) for k in ("n_steps", "batch_size", "n_epochs", "learning_rate",
                                             "gamma", "gae_lambda", "ent_coef")}
    if args.resume and (args.linear_lr or any(v is not None for v in ppo_args.values())):
        p.error("Les hyperparamètres PPO ne s'appliquent qu'à un nouveau réseau, pas à une reprise")
    if (args.train_demands is None) != (args.validation_demands is None):
        p.error("--train-demands et --validation-demands vont ensemble")
    if args.per_demand_reward and args.train_demands is None:
        p.error("--per-demand-reward exige --train-demands")
    if args.coordinated and (args.resume or args.action_mode != "duration"):
        p.error("--coordinated concerne un nouveau réseau en mode duration")
    if args.neutral_init and (args.resume or args.action_mode != "duration"):
        p.error("--neutral-init ne concerne qu'un nouveau réseau en mode duration")
    if args.train_demands:
        try:
            train_demands = load_demands(args.train_demands, "train")
            validation_demands = load_demands(args.validation_demands, "validation")
            check_disjoint(train_demands, validation_demands)
        except ValueError as error:
            p.error(str(error))
    else:
        train_demands = validation_demands = None
    config = emission_protocol(yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8")))
    args.weights = args.weights if args.weights is not None else config.get("emission_rl", {}).get("weights", [1,1,1,1])
    if args.seconds < 10 or args.timesteps < 32 or args.validation_every < 64 or set(args.train_seeds)&set(args.validation_seeds):
        p.error("Durée >=10, timesteps >=32 et seeds entraînement/validation distinctes requis")
    if not all(np.isfinite(w) and w>=0 for w in args.weights) or not any(args.weights):
        p.error("Poids finis positifs ou nuls, au moins un strictement positif")
    output = (args.output or MODEL_DIR).resolve()
    if output.exists() and any(output.iterdir()):
        p.error("Le dossier de sortie doit être nouveau ou vide (préserve les modèles existants)")
    output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(1)
    net, route = SCENARIOS[args.scenario]
    config["sim"].update(scenario=args.scenario, net_file=net, route_file=route, seconds=args.seconds)
    safety = training_safety(config)
    print("Calibration FixedTime sur seeds d’entraînement uniquement…", flush=True)
    layout, scales = calibrate(config, args.train_seeds, train_demands)
    manifest = {"schema":1, "simulation":simulation_protocol(), "scenario":args.scenario, "topology":layout, "scales_mg_s":scales,
                "weights":args.weights, "safety":safety, "train_seeds":args.train_seeds,
                "validation_seeds":args.validation_seeds, "seconds":args.seconds,
                "input_hash":digest_files([ROOT/net,ROOT/route]),
                "qualification":"smoke_test" if args.timesteps<10000 else "experimental_long" if args.timesteps>=100000 else "experimental",
                "reward":"negative weighted quantities / fixed training reference rates",
                "parent_model":str(args.resume.resolve()) if args.resume else None}
    manifest.update(action_mode=args.action_mode, reward_mode=args.reward_mode, deterministic_inference=True)
    if train_demands:
        # input_hash keeps the scenario route so evaluation and dashboard still accept the model.
        manifest["demands"] = {"train": describe(train_demands), "validation": describe(validation_demands),
                               "sampling_seed": args.demand_seed, "calibration_files": min(10, len(train_demands))}
        if args.per_demand_reward:
            print("Références FixedTime par demande d’entraînement…", flush=True)
            # Validation demands keep the global scales: no validation data in the reward.
            manifest["demand_scales_mg_s"] = demand_scales(config, args.train_seeds[0], train_demands, args.workers)
            manifest["demands"]["reward_scales"] = f"FixedTime per training demand, seed {args.train_seeds[0]}"
    if args.green_durations and (args.action_mode != "duration" or args.resume):
        p.error("--green-durations concerne un nouveau réseau en mode duration")
    if args.action_mode == "duration":
        manifest["green_durations_s"] = sorted(set(args.green_durations or [20,25,30,35]))
        manifest["coordinated_action"] = args.coordinated
        durations = manifest["green_durations_s"]
        if len(durations) < 2 or safety["min_green"] > durations[0] or safety["max_green"] < durations[-1]:
            p.error("Les durées expérimentales ne respectent pas les protections")
    if args.reward_mode == "pm_tolerant":
        if not (0 <= args.pm_tolerance < 1 and args.pm_penalty >= 0):
            p.error("Tolérance PM dans [0, 1) et pénalité positive requises")
        manifest["reward"] = "sum of per-second negative mean CO2/NOx/fuel ratio minus penalty x PM excess over 1 + tolerance; classical FixedTime rates"
        manifest.update(pm_tolerance=args.pm_tolerance, pm_penalty=args.pm_penalty)
    if args.reward_mode == "balanced":
        manifest["reward"] = "potential difference: half worst normalized cumulative emission + half weighted mean; fixed training rates, x100"
    if args.reward_mode == "worst_interval":
        manifest["reward"] = "sum of per-second negative half mean plus half worst normalized emission rate plus squared excess; equal four metrics; gamma=1"
        manifest["weights"] = [1., 1., 1., 1.]
    if args.resume:
        parent=json.loads((args.resume/"manifest.json").read_text(encoding="utf-8"))
        if parent.get("action_mode", "phase") != args.action_mode or (parent.get("reward_mode", "weighted") != args.reward_mode and not args.adapt_reward):
            p.error("Actions ou récompense du parent incompatibles avec cette reprise")
        if parent["topology"]!=layout or parent["scenario"]!=args.scenario:
            p.error("Modèle parent incompatible avec le réseau")
        if set(args.train_seeds)&set(parent["validation_seeds"]) or set(args.validation_seeds)&set(parent["train_seeds"]):
            p.error("Les seeds de la lignée du modèle doivent rester séparées entre entraînement et validation")
        parent_demands = parent.get("demands", {})
        lineage = lambda role: {e["sha256"] for e in parent_demands.get(role, [])} | {e["sha256"] for e in manifest.get("demands", {}).get(role, [])}
        if lineage("train") & lineage("validation"):
            p.error("Les demandes de la lignée du modèle doivent rester séparées entre entraînement et validation")
        # Recalibrate when the physical integration protocol changes.
        if parent.get("simulation") == manifest["simulation"]:
            manifest["scales_mg_s"] = parent["scales_mg_s"]
        else:
            # Actor inputs must retain their original meaning on warm start.
            # Reward normalizers may change without changing observations.
            manifest["observation_scales_mg_s"] = parent.get("observation_scales_mg_s", parent["scales_mg_s"])
        manifest["parent_model_hash"] = digest_files([args.resume/"best.zip", args.resume/"manifest.json"])
        manifest["train_seeds"] = sorted(set(args.train_seeds + parent["train_seeds"]))
        manifest["validation_seeds"] = sorted(set(args.validation_seeds + parent["validation_seeds"]))
    atomic_json(output/"manifest.json",manifest)
    env = EmissionTrainingEnv(config,manifest,args.train_seeds,train_demands,True,args.demand_seed)
    worst = args.reward_mode == "worst_interval"
    hyper = {"n_steps": args.n_steps or 64, "batch_size": args.batch_size or 32, "n_epochs": args.n_epochs or 4,
             "learning_rate": args.learning_rate or (1e-4 if worst else 3e-4),
             "gamma": args.gamma if args.gamma is not None else (1. if worst else .99),
             "gae_lambda": args.gae_lambda if args.gae_lambda is not None else .95,
             "ent_coef": args.ent_coef or 0.}
    if hyper["n_steps"] % hyper["batch_size"] or not 0 < hyper["gamma"] <= 1 or hyper["learning_rate"] <= 0:
        p.error("n_steps multiple de batch_size, 0 < gamma <= 1 et learning_rate > 0 requis")
    model = PPO.load(args.resume/"best.zip", env=env,device="cpu") if args.resume else PPO(
        "MlpPolicy",env,**hyper,seed=args.train_seeds[0],device="cpu",verbose=0)
    if not args.resume:
        manifest["ppo"] = {**hyper, "linear_lr": args.linear_lr}
        if args.linear_lr:
            # SB3 progress restarts with every validation block; decay on the
            # whole requested budget instead.
            base_lr, budget_steps = hyper["learning_rate"], args.timesteps
            model.lr_schedule = lambda _: base_lr * max(0., 1. - model.num_timesteps / budget_steps)
    if worst and args.resume:
        model.gamma = 1.
        model.rollout_buffer.gamma = 1.
        model.learning_rate = 1e-4
        model.lr_schedule = lambda progress: 1e-4
        # The old value function estimates a different reward. Reset its head
        # and optimizer momentum while retaining the actor's learned policy.
        with torch.no_grad():
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.zero_()
        model.policy.optimizer.state.clear()
        manifest.update(gamma=1., learning_rate=1e-4, optimizer_reset=True)
    elif worst:
        manifest.update(gamma=hyper["gamma"], learning_rate=hyper["learning_rate"], optimizer_reset=False)
    if not args.resume and args.action_mode == "duration":
        init_duration_policy(model.policy, args.neutral_init, manifest["green_durations_s"])
        manifest["initial_policy"] = "uniform" if args.neutral_init else "prefer_30s"
    print("Références feux classiques sur seeds de validation uniquement…", flush=True)
    baseline = validation_baseline(config, args.validation_seeds, output, validation_demands)
    target = None
    if args.target_green:
        target_config = copy.deepcopy(config)
        target_config["fixed_time"]["green"] = args.target_green
        target = validation_baseline(target_config, args.validation_seeds, output, validation_demands, "validation_target")
        manifest["validation_target_green_s"] = args.target_green
    callback=Validate(env,config,manifest,args.validation_seeds,output,max(64,args.timesteps),baseline=baseline,
                      demands=validation_demands,target=target)
    try:
        callback.init_callback(model)
        start_steps=model.num_timesteps
        if args.resume or args.action_mode == "duration":
            print("Validation du point de départ avant apprentissage…", flush=True)
            callback.num_timesteps = start_steps
            callback.assess()
            model.set_env(model.get_env(), force_reset=True)
        while model.num_timesteps-start_steps < args.timesteps:
            budget=min(args.validation_every,args.timesteps-(model.num_timesteps-start_steps))
            print(f"Apprentissage PPO : bloc de {budget} pas…",flush=True)
            model.learn(total_timesteps=budget,reset_num_timesteps=False,callback=callback)
            print("Validation sur seeds distinctes…",flush=True)
            callback.num_timesteps=model.num_timesteps
            callback.assess()
            model.set_env(model.get_env(), force_reset=True)
        model.save(output/"last.zip")
        manifest["actual_timesteps"]=model.num_timesteps
        manifest["new_timesteps"]=model.num_timesteps-start_steps
        manifest["best_validation_reward"]=callback.best
        manifest["best_timesteps"]=callback.best_timesteps
        manifest["best_validation_teleports"]=callback.best_teleports
        manifest["best_validation_collisions"]=callback.best_collisions
        manifest["best_validation_comparison"] = callback.best_comparison
        manifest["selection"]="minimize collisions, teleports, worst paired validation ratio to FixedTime (four emissions and traffic shortfall), then weighted emission cost"
        if args.reward_mode == "pm_tolerant":
            manifest["selection"]="minimize collisions, teleports; then PM within tolerance, no arrival loss, at most 1 departure fewer; then lowest mean CO2/NOx/fuel ratio to classical FixedTime"
        atomic_json(output/"manifest.json",manifest)
    finally:
        env.close()
    print(f"Modèle sauvegardé : {output}",flush=True)


if __name__=="__main__":
    main()
