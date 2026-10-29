"""Initialize the emission PPO policy by imitating the rules agent (behaviour cloning + DAgger).

The policy proposes a phase per junction (action mode "phase"); the SafetyGate
keeps the trained safeguards. The teacher is the rules agent with the same
safeguards. Rounds after the first let the policy drive and the teacher label
the states it reaches (DAgger), so the policy learns to recover from its own
mistakes. Each round is checked on the validation demands against FixedTime
(official tripinfo totals); the best feasible round becomes best.zip.

    python scripts/imitate_rules.py --output runs/models/emission_imitation
    python scripts/imitate_rules.py --evaluate runs/models/emission_imitation --pool test --seeds 10200 10201
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import yaml
from flowmind.env import ROOT, SumoCorridorEnv, simulation_protocol
from flowmind.agents.baselines import FixedTime
from flowmind.agents.rules import RuleBasedAgent
from flowmind.emission_rl import SafetyGate, decode_actions, encode, training_safety
from flowmind.evaluation import SCENARIOS, digest_files, emission_protocol
from flowmind.demands import DEMAND_DIR, check_disjoint, describe, load_demands, validation_cases
from flowmind.kpis import atomic_json
from scripts.train_emissions import (EmissionTrainingEnv, calibrate, compare_validation, official_emissions,
                                     tolerant_selection)

CONTROLLERS = ("fixed20", "rules", "teacher", "model")


def load_config(seconds, scenario="3x3grid"):
    config = emission_protocol(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")))
    net, route = SCENARIOS[scenario]
    config["sim"].update(scenario=scenario, net_file=net, route_file=route, seconds=seconds)
    return config


def rollout(job):
    """One episode: observations and teacher labels; the learner drives when given."""
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    config, manifest, route, seed, learner = job
    ids = [item["id"] for item in manifest["topology"]]
    samples, observations, labels = [], [], []
    model = PPO.load(learner, device="cpu") if learner else None
    teacher = {k: RuleBasedAgent(manifest["safety"]) for k in ids}
    gate = SafetyGate(manifest["safety"])
    env = SumoCorridorEnv(config=copy.deepcopy(config), route_file=route, seed=seed, use_gui=False)
    env.metrics_callback = samples.append
    try:
        states = env.reset()
        while samples[-1]["elapsed_s"] < env.sim_seconds:
            obs = encode(states, manifest)
            label = [teacher[k].act(states[k], [], [True] * states[k].num_phases) for k in ids]
            if model is None:
                proposed = dict(zip(ids, label))
            else:
                proposed = decode_actions(states, model.predict(obs, deterministic=True)[0], manifest)
            observations.append(obs)
            labels.append(label)
            states, _, _ = env.step(gate.apply(states, proposed)[0])
    finally:
        env.close()
    return np.asarray(observations, np.float32), np.asarray(labels, np.int64)


def run_case(job):
    """Whole-window official totals of one controller on one (demand, seed) case."""
    import torch
    torch.set_num_threads(1)
    config, controller, route, seed, directory, model_dir = job
    from flowmind.emission_rl import EmissionPolicy
    config = copy.deepcopy(config)
    config["evaluation_output"] = str(directory)
    rules = config["rules"] | {k: config["signals"][k] for k in ("min_green", "max_green")}
    samples = []
    env = SumoCorridorEnv(config=config, route_file=route, seed=seed, use_gui=False)
    env.metrics_callback = samples.append
    try:
        states = env.reset()
        if controller == "model":
            policy = EmissionPolicy(config["sim"]["scenario"], rules, directory=model_dir, seed=seed)
        else:
            params = training_safety(config) if controller == "teacher" else rules
            agents = {k: FixedTime(config["fixed_time"]) if controller == "fixed20" else RuleBasedAgent(params)
                      for k in states}
        while samples[-1]["elapsed_s"] < env.sim_seconds:
            if controller == "model":
                actions = policy.act_all(states)[0]
            else:
                actions = {k: a.act(states[k], [], [True] * states[k].num_phases) for k, a in agents.items()}
            states, _, _ = env.step(actions)
    finally:
        env.close()
    result = official_emissions(Path(directory) / "tripinfo.xml")
    result.update(teleports=sum(s["teleports"] for s in samples), collisions=sum(s["collisions"] for s in samples))
    return result


def compare(config, controllers, cases, output, workers, model_dir=None):
    """{controller: {case key: totals}} for every (seed, route) case."""
    jobs, keys = [], []
    for controller in controllers:
        for seed, route in cases:
            key = f"{Path(route).name.removesuffix('.rou.xml')}_seed{seed}"
            jobs.append((config, controller, str(route), seed, output / controller / key, model_dir))
            keys.append((controller, key))
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(run_case, jobs))
    table = {c: {} for c in controllers}
    for (controller, key), result in zip(keys, results):
        table[controller][key] = result
    return table


def fit(model, observations, labels, epochs, learning_rate, batch_size=256, seed=0):
    """Supervised cross-entropy on the policy head; returns per-junction accuracy."""
    import torch
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    obs = torch.as_tensor(observations)
    act = torch.as_tensor(labels)
    generator = torch.Generator().manual_seed(seed)
    policy.train()
    for _ in range(epochs):
        for index in torch.randperm(len(obs), generator=generator).split(batch_size):
            loss = -policy.get_distribution(obs[index]).log_prob(act[index]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return accuracy(model, observations, labels)


def accuracy(model, observations, labels):
    import torch
    with torch.no_grad():
        dist = model.policy.get_distribution(torch.as_tensor(observations)).distribution
        choice = torch.stack([d.probs.argmax(1) for d in dist], 1)
    return float((choice.numpy() == labels).mean())


def summary(table, reference):
    """Paired ratios of every controller to the reference controller."""
    return {c: compare_validation(rows, table[reference]) for c, rows in table.items() if c != reference}


def print_summary(title, ratios):
    print(title, flush=True)
    for controller, r in ratios.items():
        print(f"  {controller:8s} CO2 {100*(r['co2_ratio']-1):+6.2f}%  NOx {100*(r['nox_ratio']-1):+6.2f}%  "
              f"carburant {100*(r['fuel_ratio']-1):+6.2f}%  PM {100*(r['pm_ratio']-1):+6.2f}%  "
              f"arrivées manquantes {100*(r['arrival_shortfall_ratio']-1):+5.2f}%", flush=True)


def train(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        sys.exit("Le dossier de sortie doit être nouveau ou vide (préserve les modèles existants)")
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from stable_baselines3 import PPO
    torch.manual_seed(args.seed)
    config = load_config(args.seconds)
    train_demands = load_demands(DEMAND_DIR / "train", "train")
    validation_demands = load_demands(DEMAND_DIR / "validation", "validation")
    check_disjoint(train_demands, validation_demands)
    safety = training_safety(config)
    print("Calibration des échelles d’observation (FixedTime, demandes d’entraînement)…", flush=True)
    layout, scales = calibrate(config, args.train_seeds, train_demands)
    net, route = SCENARIOS["3x3grid"]
    manifest = {"schema": 1, "simulation": simulation_protocol(), "scenario": "3x3grid", "topology": layout,
                "scales_mg_s": scales, "weights": config.get("emission_rl", {}).get("weights", [1, 1, 1, 1]),
                "safety": safety, "train_seeds": args.train_seeds, "validation_seeds": args.validation_seeds,
                "seconds": args.seconds, "input_hash": digest_files([ROOT / net, ROOT / route]),
                "action_mode": "phase", "reward_mode": "imitation", "training_method": "imitation",
                "deterministic_inference": True, "qualification": "experimental",
                "reward": "none: behaviour cloning of the rules agent (same safeguards), DAgger rounds",
                "demands": {"train": describe(train_demands), "validation": describe(validation_demands)},
                "parent_model": None}
    env = EmissionTrainingEnv(config, manifest, args.train_seeds)
    model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", policy_kwargs={"net_arch": [256, 256]})
    train_cases = [(seed, str(r)) for seed, r in validation_cases(args.train_seeds, train_demands)]
    validation = [(seed, str(r)) for seed, r in validation_cases(args.validation_seeds, validation_demands)]
    print("Références de validation : feux fixes, règles, règles avec les protections du modèle…", flush=True)
    reference = compare(config, ("fixed20", "rules", "teacher"), validation, output / "validation", args.workers)
    print_summary("Références (par rapport aux feux fixes) :", summary(reference, "fixed20"))
    held_out = [(config, manifest, r, s, None) for s, r in validation]
    with ProcessPoolExecutor(args.workers) as pool:
        test_obs, test_labels = map(np.concatenate, zip(*pool.map(rollout, held_out)))
    observations, labels, rounds, best = [], [], [], None
    for round_index in range(args.rounds + 1):
        learner = None if round_index == 0 else str(output / f"round_{round_index - 1}.zip")
        with ProcessPoolExecutor(args.workers) as pool:
            for obs, lab in pool.map(rollout, [(config, manifest, r, s, learner) for s, r in train_cases]):
                observations.append(obs)
                labels.append(lab)
        data_obs, data_labels = np.concatenate(observations), np.concatenate(labels)
        fit_accuracy = fit(model, data_obs, data_labels, args.epochs, args.learning_rate, seed=args.seed + round_index)
        held_accuracy = accuracy(model, test_obs, test_labels)
        path = output / f"round_{round_index}.zip"
        model.save(path)
        candidate = output / f"candidate_{round_index}"
        candidate.mkdir()
        shutil.copy(path, candidate / "best.zip")
        atomic_json(candidate / "manifest.json", manifest)
        table = compare(config, ("model",), validation, output / "validation" / f"round_{round_index}",
                        args.workers, candidate)
        ratios = compare_validation(table["model"], reference["fixed20"])
        versus_rules = compare_validation(table["model"], reference["rules"])
        feasible, score = tolerant_selection(ratios, args.pm_tolerance)
        collisions = sum(r["collisions"] + r["teleports"] for r in table["model"].values())
        row = {"round": round_index, "samples": int(len(data_obs)), "train_accuracy": fit_accuracy,
               "validation_accuracy": held_accuracy, "feasible": feasible and collisions == 0,
               "versus_fixed20": ratios, "versus_rules": versus_rules, "collisions_teleports": collisions}
        rounds.append(row)
        atomic_json(output / "rounds.json", {"rounds": rounds})
        print(f"Tour {round_index} : {len(data_obs)} exemples, précision entraînement {fit_accuracy:.3f}, "
              f"validation {held_accuracy:.3f}", flush=True)
        print_summary("  Modèle par rapport aux feux fixes / aux règles :",
                      {"vs fixes": ratios, "vs règles": versus_rules})
        key = (row["feasible"], score)
        if best is None or key > best[0]:
            best = (key, round_index)
        shutil.rmtree(candidate)
    chosen = rounds[best[1]]
    shutil.copy(output / f"round_{best[1]}.zip", output / "best.zip")
    manifest.update(actual_timesteps=0, best_timesteps=0, imitation_round=best[1], imitation_samples=chosen["samples"],
                    best_validation_comparison=chosen["versus_fixed20"],
                    best_validation_versus_rules=chosen["versus_rules"],
                    selection=f"feasible (PM within +{args.pm_tolerance:.0%}, no traffic loss, no collision), "
                              "then lowest mean CO2/NOx/fuel ratio to FixedTime on validation demands",
                    validation_reference=f"FixedTime {config['fixed_time']['green']} s")
    atomic_json(output / "manifest.json", manifest)
    print(f"Tour retenu : {best[1]} -> {output / 'best.zip'}", flush=True)


def evaluate(args):
    config = load_config(args.seconds)
    model_dir = args.evaluate.resolve()
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if set(args.seeds) & set(manifest["train_seeds"] + manifest["validation_seeds"]):
        sys.exit("Seeds de test distinctes de l’entraînement et de la validation requises")
    demands = load_demands(DEMAND_DIR / args.pool, args.pool)
    cases = [(seed, str(route)) for route in demands for seed in args.seeds]
    output = ROOT / "runs/reports" / f"{model_dir.name}_{args.pool}"
    if output.exists():
        sys.exit(f"Rapport déjà présent : {output}")
    table = compare(config, ("fixed20", "rules", "model"), cases, output, args.workers, model_dir)
    ratios = summary(table, "fixed20")
    ratios["model vs rules"] = compare_validation(table["model"], table["rules"])
    per_case = {key: {c: table[c][key] for c in table} for key in table["fixed20"]}
    atomic_json(output / "report.json", {"model": str(model_dir), "pool": args.pool, "seeds": args.seeds,
                                         "ratios": ratios, "cases": per_case})
    print_summary(f"{len(cases)} cas ({args.pool}), par rapport aux feux fixes {config['fixed_time']['green']} s :", ratios)
    model_co2 = [table["model"][k]["emissions_mg"][0] / table["fixed20"][k]["emissions_mg"][0] - 1 for k in per_case]
    print(f"  Cas où le modèle réduit le CO2 : {sum(v < 0 for v in model_co2)}/{len(model_co2)} ; "
          f"collisions+téléportations : {sum(r['collisions'] + r['teleports'] for r in table['model'].values())}")
    print(f"Rapport : {output / 'report.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=ROOT / "runs/models/emission_imitation")
    p.add_argument("--evaluate", type=Path, help="Evaluate a saved model directory instead of training")
    p.add_argument("--pool", choices=["validation", "test"], default="test")
    p.add_argument("--seeds", nargs="+", type=int, default=[10200, 10201], help="--evaluate: SUMO seeds")
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--validation-seeds", nargs="+", type=int, default=[1000, 1001])
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument("--rounds", type=int, default=3, help="DAgger rounds after the first behaviour cloning")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--pm-tolerance", type=float, default=.02)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if set(args.train_seeds) & set(args.validation_seeds):
        p.error("Seeds d’entraînement et de validation distinctes requises")
    if args.rounds < 0 or args.epochs < 1 or args.seconds < 10:
        p.error("--rounds >= 0, --epochs >= 1, --seconds >= 10")
    evaluate(args) if args.evaluate else train(args)


if __name__ == "__main__":
    main()
