"""Train/fine-tune an emission-only PPO policy on SUMO, with held-out validation."""
from __future__ import annotations
import argparse
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
from flowmind.env import ROOT, SumoCorridorEnv
from flowmind.agents.baselines import FixedTime
from flowmind.emission_rl import MODEL_DIR, METRICS, SAMPLE_KEYS, SafetyGate, encode, topology, training_safety
from flowmind.evaluation import SCENARIOS, digest_files
from flowmind.kpis import atomic_json, write_csv


class EmissionTrainingEnv(gym.Env):
    def __init__(self, config, manifest, seeds):
        super().__init__()
        self.config, self.manifest, self.seeds = config, manifest, seeds
        self.index = 0
        self.env = None
        self.samples, self.episode_rows = [], []
        self.action_space = gym.spaces.MultiDiscrete([t["phases"] for t in manifest["topology"]])
        size = sum(t["phases"] * 5 + 1 for t in manifest["topology"])
        self.observation_space = gym.spaces.Box(0, np.inf, (size,), np.float32)
        self.total_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        episode_seed = self.seeds[self.index % len(self.seeds)]
        self.index += 1
        self.samples = []
        self.env = SumoCorridorEnv(config=copy.deepcopy(self.config), seed=episode_seed, use_gui=False)
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
        proposed = dict(zip((t["id"] for t in self.manifest["topology"]), np.asarray(action).reshape(-1)))
        safe, _ = self.gate.apply(self.states, proposed)
        self.samples.clear()
        self.states, _, info = self.env.step(safe)
        totals = [sum(s[k] for s in self.samples) for k in SAMPLE_KEYS]
        self.episode_quantities = [a+b for a,b in zip(self.episode_quantities, totals)]
        for key in self.episode_stats:
            self.episode_stats[key] += sum(s[key] for s in self.samples)
        reward = environmental_reward(totals, self.manifest["scales_mg_s"], self.manifest["weights"])
        self.total_reward += reward
        self.steps += 1
        truncated = info["time"] - self.env.begin_time >= self.env.sim_seconds
        if truncated:
            self.episode_rows.append({"episode": len(self.episode_rows)+1, "seed": self.episode_seed,
                                      "reward": self.total_reward, "decisions": self.steps})
        return encode(self.states, self.manifest), reward, False, truncated, {}

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


def calibrate(config, seeds):
    samples, layout = [], None
    for seed in seeds:
        env = SumoCorridorEnv(config=copy.deepcopy(config), seed=seed, use_gui=False)
        env.metrics_callback = samples.append
        try:
            states = env.reset()
            layout = topology(states)
            agents = {k: FixedTime(config["fixed_time"]) for k in states}
            while samples[-1]["elapsed_s"] < env.sim_seconds:
                states, _, _ = env.step({k:a.act(states[k], [], [True]*states[k].num_phases) for k,a in agents.items()})
        finally:
            env.close()
    # Fixed reference rates; floors avoid division by zero for absent pollutants.
    scales = [max(1.0, sum(s[k] for s in samples)/len(samples)) for k in SAMPLE_KEYS]
    return layout, scales


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


def validation_baseline(config, seeds, output):
    """Reference used for model selection, never for learning or normalization."""
    result = {}
    for seed in seeds:
        directory = output / "validation_baseline" / f"seed_{seed}"
        cfg = copy.deepcopy(config)
        cfg["evaluation_output"] = str(directory)
        env = SumoCorridorEnv(config=cfg, seed=seed, use_gui=False)
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
        result[seed] = official_emissions(directory / "tripinfo.xml")
        print(f"Référence de validation : seed={seed}, arrivées={result[seed]['arrived']}", flush=True)
    atomic_json(output / "validation_baseline.json", {"seeds": result})
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
            "worst_ratio": max(1., *ratios, arrivals, departures),
            "all_emissions_lower": all(r < 1 for r in ratios)}


class Validate(BaseCallback):
    def __init__(self, raw, config, manifest, seeds, output, every, baseline=None):
        super().__init__()
        self.raw, self.config, self.manifest = raw, config, manifest
        self.seeds, self.output, self.every = seeds, output, every
        self.best, self.rows = -np.inf, []
        self.best_timesteps = None
        self.best_key = None
        self.best_teleports = self.best_collisions = None
        self.baseline = baseline
        self.best_comparison = None

    def assess(self):
        # Training and validation share the default TraCI connection: close first.
        self.raw.close()
        validation = EmissionTrainingEnv(self.config, self.manifest, self.seeds)
        scores, diagnostics, quantities, measured = [], [], [], {}
        try:
            for seed in self.seeds:
                if self.baseline is not None:
                    directory = self.output / "validation_trials" / f"step_{self.num_timesteps}" / f"seed_{seed}"
                    validation.config = copy.deepcopy(self.config)
                    validation.config["evaluation_output"] = str(directory)
                obs, _ = validation.reset()
                finished = False
                while not finished:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, _, finished, _ = validation.step(action)
                scores.append(validation.total_reward)
                diagnostics.append(getattr(validation, "episode_stats", {}))
                quantities.append(getattr(validation, "episode_quantities", [0.] * 4))
                if self.baseline is not None:
                    validation.env.close()
                    measured[seed] = official_emissions(directory / "tripinfo.xml")
        finally:
            validation.close()
        score = float(np.mean(scores))
        totals = {k: sum(d.get(k, 0) for d in diagnostics) for k in ("teleports", "collisions", "departed", "arrived")}
        comparison = compare_validation(measured, self.baseline) if self.baseline is not None else {}
        self.rows.append({"timesteps": self.num_timesteps, "validation_reward": score, **totals,
                          **{f"{name}_mg": float(np.mean([q[i] for q in quantities])) for i,name in enumerate(METRICS)},
                          **comparison})
        write_csv(self.output / "validation.csv", self.rows)
        write_csv(self.output / "training_episodes.csv", self.raw.episode_rows)
        selection_key = (-totals["collisions"], -totals["teleports"],
                         -comparison.get("worst_ratio", 1.), score)
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
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    config = yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8"))
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
    layout, scales = calibrate(config, args.train_seeds)
    manifest = {"schema":1, "scenario":args.scenario, "topology":layout, "scales_mg_s":scales,
                "weights":args.weights, "safety":safety, "train_seeds":args.train_seeds,
                "validation_seeds":args.validation_seeds, "seconds":args.seconds,
                "input_hash":digest_files([ROOT/net,ROOT/route]),
                "qualification":"smoke_test" if args.timesteps<10000 else "experimental",
                "reward":"negative weighted quantities / fixed training reference rates",
                "parent_model":str(args.resume.resolve()) if args.resume else None}
    if args.resume:
        parent=json.loads((args.resume/"manifest.json").read_text(encoding="utf-8"))
        if parent["topology"]!=layout or parent["scenario"]!=args.scenario:
            p.error("Modèle parent incompatible avec le réseau")
        if set(args.train_seeds)&set(parent["validation_seeds"]) or set(args.validation_seeds)&set(parent["train_seeds"]):
            p.error("Les seeds de la lignée du modèle doivent rester séparées entre entraînement et validation")
        manifest["scales_mg_s"] = parent["scales_mg_s"]
        manifest["parent_model_hash"] = digest_files([args.resume/"best.zip", args.resume/"manifest.json"])
        manifest["train_seeds"] = sorted(set(args.train_seeds + parent["train_seeds"]))
        manifest["validation_seeds"] = sorted(set(args.validation_seeds + parent["validation_seeds"]))
    atomic_json(output/"manifest.json",manifest)
    env = EmissionTrainingEnv(config,manifest,args.train_seeds)
    model = PPO.load(args.resume/"best.zip", env=env,device="cpu") if args.resume else PPO(
        "MlpPolicy",env,n_steps=64,batch_size=32,n_epochs=4,seed=args.train_seeds[0],device="cpu",verbose=0)
    print("Références feux classiques sur seeds de validation uniquement…", flush=True)
    baseline = validation_baseline(config, args.validation_seeds, output)
    callback=Validate(env,config,manifest,args.validation_seeds,output,max(64,args.timesteps),baseline=baseline)
    try:
        callback.init_callback(model)
        start_steps=model.num_timesteps
        if args.resume:
            print("Validation du modèle parent avant apprentissage…", flush=True)
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
        atomic_json(output/"manifest.json",manifest)
    finally:
        env.close()
    print(f"Modèle sauvegardé : {output}",flush=True)


if __name__=="__main__":
    main()
