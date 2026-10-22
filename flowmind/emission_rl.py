"""Shared PPO observation encoding and safety for training and frozen inference."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from flowmind.agents.rules import RuleBasedAgent

METRICS = ("co2", "nox", "fuel", "pm")
SAMPLE_KEYS = ("co2_mg_s", "nox_mg_s", "fuel_mg_s", "pmx_mg_s")
MODEL_DIR = Path(__file__).resolve().parent.parent / "runs/models/emission_ppo"


def training_safety(config):
    """RL-specific limits; leave the shared environment and reference unchanged."""
    params = config["rules"] | {k: config["signals"][k] for k in ("min_green", "max_green")}
    params.update(config.get("emission_rl", {}).get("safety", {}))
    # A decision can include yellow, all-red and box clearance at other junctions.
    reserve = config["sim"]["delta_time"] + sum(config["signals"].get(k, d) for k, d in
                                                (("yellow", 3), ("all_red", 2), ("max_red_clearance", 4)))
    if not 0 < params["min_green"] <= params["max_green"]:
        raise ValueError("Durées de vert RL incompatibles")
    if params["max_red"] <= 0 or max(params["max_red"], params["max_green"]) + reserve >= 60:
        raise ValueError("Les protections RL doivent intervenir avant le seuil SUMO de 60 s, transitions comprises")
    return params


def inference_safety(manifest, requested):
    """A dashboard command may tighten, but cannot relax the trained safeguards."""
    saved = manifest["safety"]
    result = dict(saved)
    for key in ("max_red", "max_green", "blocked_occupancy"):
        if key in requested:
            result[key] = min(saved[key], requested[key])
    if "min_green" in requested:
        result["min_green"] = max(saved["min_green"], requested["min_green"])
    if result["min_green"] > result["max_green"]:
        raise ValueError("Le vert demandé est incompatible avec les protections du modèle RL")
    return result


def topology(states):
    return [{"id": key, "phases": state.num_phases,
             "phase_lanes": [list(lanes) for lanes in state.phase_lanes]}
            for key, state in sorted(states.items())]


def encode(states, manifest):
    if topology(states) != manifest["topology"]:
        raise ValueError("Le réseau / les phases ne correspondent pas au modèle d’émissions")
    features = []
    for item in manifest["topology"]:
        state = states[item["id"]]
        rates = {r.lane_id: r for r in state.lane_emissions}
        if any(l.lane_id not in rates for l in state.lanes):
            raise ValueError("Mesures d’émissions manquantes")
        for phase in state.phase_lanes:
            features.extend(np.log1p(sum(getattr(rates[l], k) for l in set(phase) if l in rates) / scale)
                            for k, scale in zip(METRICS, manifest["scales_mg_s"]))
        features.extend(float(i == state.phase) for i in range(state.num_phases))
        features.append(min(state.time_in_phase / 300, 1))
    result = np.asarray(features, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Observation d’émissions invalide")
    return result


class SafetyGate:
    """Use the existing rule agent constraints, never its emission score as policy."""
    def __init__(self, params):
        self.params = params
        self.guards = {}

    def apply(self, states, proposed):
        actions, reasons = {}, {}
        for key, st in states.items():
            guard = self.guards.setdefault(key, RuleBasedAgent(self.params))
            # update starvation memory and obtain the safety-constrained fallback
            fallback = guard.act(st, [], [True] * st.num_phases)
            desired = int(proposed[key])
            safe = 0 <= desired < st.num_phases
            exits = st.phase_out_lanes[desired] if safe and desired < len(st.phase_out_lanes) else ()
            out = {l.lane_id: l for l in st.out_lanes}
            known = [out[l] for l in exits if l in out]
            safe = safe and not (known and all(l.occupancy >= guard.blocked_occupancy for l in known))
            overdue = any(st.time - since >= guard.max_red for since in guard.waiting_since.values())
            if st.in_transition or st.time_in_phase < guard.min_green:
                final = st.phase
            elif not safe or overdue or (st.time_in_phase >= guard.max_green and desired == st.phase):
                final = fallback
            else:
                final = desired
            actions[key] = final
            reasons[key] = f"PPO émissions : phase proposée {desired}, appliquée {final}" + (" ; contrainte de circulation" if final != desired else "")
        return actions, reasons


class EmissionPolicy:
    def __init__(self, scenario, params, directory=MODEL_DIR, seed=None):
        from stable_baselines3 import PPO
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["scenario"] != scenario:
            raise ValueError("Le modèle d’émissions a été entraîné pour un autre scénario")
        self.model = PPO.load(directory / "best.zip", device="cpu")
        self.deterministic = self.manifest.get("deterministic_inference", True)
        self.policy_seed = None
        if not self.deterministic:
            if seed is None:
                raise ValueError("Une seed est requise pour reproduire les décisions probabilistes du modèle")
            self.policy_seed = int(seed) + int(self.manifest.get("policy_seed_offset", 100000))
            self.model.set_random_seed(self.policy_seed)
        self.gate = SafetyGate(inference_safety(self.manifest, params))

    def act_all(self, states):
        predictions, _ = self.model.predict(encode(states, self.manifest), deterministic=self.deterministic)
        proposed = dict(zip((item["id"] for item in self.manifest["topology"]), np.asarray(predictions).reshape(-1)))
        return self.gate.apply(states, proposed)
