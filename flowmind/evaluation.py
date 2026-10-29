"""Evaluation orchestration. SUMO access stays inside env.py."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
import yaml
from flowmind.agents.baselines import FixedTime, MaxPressure
from flowmind.agents.flow import FlowAgent
from flowmind.agents.rules import RuleBasedAgent
from flowmind.emission_rl import EmissionPolicy, MODEL_DIR
from flowmind.env import ROOT, SumoCorridorEnv, simulation_protocol
from flowmind.kpis import atomic_json, compare_rows, compute_kpis, write_csv

CONTROLLERS = ("orchestrated", "fixed_time", "rules", "emission_model", "max_pressure", "flow_heuristic", "flow_model")
SCENARIOS = {
    "corridor": ("scenarios/corridor/corridor.net.xml", "scenarios/corridor/corridor.rou.xml"),
    "corridor_demo": ("scenarios/corridor/corridor.net.xml", "scenarios/corridor/corridor_demo.rou.xml"),
    "3x3grid": ("scenarios/3x3grid/3x3Grid2lanes.net.xml", "scenarios/3x3grid/routes14000.rou.xml"),
    "cologne3": ("scenarios/cologne3.net.xml", "scenarios/cologne3.rou.xml"),
}
# Team scenarios run with the team's settings (normal SUMO physics, fixed_time 30 s), exactly
# like scripts/run.py; the emission protocol below stays for Manoubi's grid/Cologne studies.
TEAM_SCENARIOS = ("corridor", "corridor_demo")
# Agents of the orchestrated system, in the order shown by the dashboard (added by Adem).
TEAM_AGENTS = ("emergency", "incident", "pedestrian", "bus", "flow", "max_pressure", "operator")


def team_live(states, events, decisions, shield_changes):
    """Per-agent live figures for one decision step of the orchestrated system (added by Adem).

    ctrl_<agent>: junctions controlled by that agent now; emergency_near / buses_near: vehicles
    approaching a junction; ped_waiting / ped_wait_now_max_s: people waiting at the crossings;
    incident_warnings / spillback_warnings: coordinator warnings now; shield_changes: total so far.
    """
    live = {f"ctrl_{a}": 0 for a in TEAM_AGENTS}
    for d in decisions:
        key = f"ctrl_{d.agent}"
        if key in live:
            live[key] += 1
    specials = [v for st in states.values() for v in st.special_vehicles]
    crossings = [c for st in states.values() for c in st.crossings]
    warnings = [e for evs in events.values() for e in evs if e.kind == "incident"]
    live.update(
        emergency_near=len({v.vehicle_id for v in specials if v.kind == "emergency"}),
        buses_near=len({v.vehicle_id for v in specials if v.kind == "bus"}),
        ped_waiting=sum(c.waiting for c in crossings),
        ped_wait_now_max_s=max((c.max_wait for c in crossings), default=0.0),
        incident_warnings=sum(e.data.get("cause") == "incident" for e in warnings),
        spillback_warnings=sum(e.data.get("cause") == "spillback" for e in warnings),
        shield_changes=shield_changes,
    )
    return live


def team_kpis(tripinfo):
    """Final per-agent figures from SUMO's trip file (added by Adem). Only finished trips count
    (the dashboard asks SUMO to also write unfinished ones, with arrival = -1)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(tripinfo).getroot()
    trips = [t.attrib for t in root.iter("tripinfo") if float(t.get("arrival", -1)) >= 0]
    people = [p.attrib for p in root.iter("personinfo") if float(p.get("duration", -1)) >= 0]

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    buses = [t for t in trips if "bus" in t["id"].lower() or "bus" in t.get("vType", "").lower()]
    ambulances = {t["id"]: float(t["duration"]) for t in trips
                  if "emergency" in t.get("vType", "").lower() or "amb" in t["id"].lower()}
    return {
        "team_arrived": len(trips),
        "team_mean_waiting_s": mean([float(t["waitingTime"]) for t in trips]),
        "team_mean_trip_s": mean([float(t["duration"]) for t in trips]),
        "bus_trips": len(buses),
        "bus_mean_trip_s": mean([float(t["duration"]) for t in buses]),
        "bus_mean_waiting_s": mean([float(t["waitingTime"]) for t in buses]),
        "pedestrians": len(people),
        "ped_mean_wait_s": mean([float(p["waitingTime"]) for p in people]),
        "ped_max_wait_s": max((float(p["waitingTime"]) for p in people), default=None),
        "ambulance_trip_s": ambulances,
    }


def emission_protocol(config):
    """Settings used only by the emission experiments, never by the team's runs:
    reproducible SUMO (sim.reproducible) and the 20 s fixed-time reference
    (evaluation.fixed_time_green; the shared fixed_time.green stays 30 s)."""
    config.setdefault("sim", {})["reproducible"] = True
    green = config.get("evaluation", {}).get("fixed_time_green")
    if green is not None:
        config["fixed_time"] = {**config.get("fixed_time", {}), "green": green}
    return config


def digest_files(paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


class Recorder:
    def __init__(self, directory, metadata):
        self.directory, self.metadata = directory, metadata
        self.f = (directory / "timeseries.csv").open("w", newline="", encoding="utf-8-sig")
        self.writer = None
        self.last = {}

    def sample(self, values):
        values = values | {"controller": self.metadata["controller"],
                           "control_revision": self.metadata.get("control_revision", 0)}
        if self.writer is None:
            self.writer = csv.DictWriter(self.f, fieldnames=list(values))
            self.writer.writeheader()
        self.writer.writerow(values)
        self.f.flush()
        self.last = values
        if values["elapsed_s"] == 1 or int(values["elapsed_s"]) % 5 == 0:
            atomic_json(self.directory / "live.json", self.metadata | values)

    def close(self):
        self.f.close()


def run_experiment(controller="fixed_time", scenario="3x3grid", seconds=600,
                   seed=100, gui=False, gui_delay_ms=100, stop_event=None, runtime_control=None,
                   model_directory=None):
    if controller not in CONTROLLERS or scenario not in SCENARIOS:
        raise ValueError("Unknown controller or scenario")
    if not 10 <= int(seconds) <= 86400:
        raise ValueError("The length must be between 10 and 86400 seconds")
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    team = scenario in TEAM_SCENARIOS
    if not team:
        config = emission_protocol(config)
    if controller == "orchestrated" and runtime_control:
        raise ValueError("The full FlowMind system cannot be switched during a run")
    initial_control = runtime_control.snapshot() if runtime_control else None
    if initial_control:
        controller = initial_control["method"]
        config.setdefault("rules", {}).update(initial_control["rules"])
        for key in ("min_green", "max_green"):
            config["signals"][key] = initial_control["rules"][key]
    net, route = SCENARIOS[scenario]
    config["sim"].update(scenario=scenario, net_file=net, route_file=route, seconds=int(seconds))
    flow_params = dict(config.get("flow", {}))
    model_hash = "none"
    emission_policy = None
    emission_manifest = None
    model_directory = Path(model_directory) if model_directory is not None else MODEL_DIR
    if controller == "emission_model" or runtime_control:
        if (model_directory / "best.zip").is_file():
            model_hash = digest_files([model_directory / "best.zip", model_directory / "manifest.json"])
            emission_manifest = json.loads((model_directory / "manifest.json").read_text(encoding="utf-8"))
        elif controller == "emission_model":
            raise ValueError("Modèle émissions absent : lancer scripts/train_emissions.py")
    if controller == "emission_model" and emission_manifest:
        if seed in emission_manifest["train_seeds"] + emission_manifest["validation_seeds"]:
            raise ValueError("Seed réservée à l’entraînement/validation : choisir une seed de test distincte")
        if emission_manifest["input_hash"] != digest_files([ROOT/net,ROOT/route]):
            raise ValueError("Le réseau ou le trafic a changé depuis l’entraînement du modèle")
    if controller == "flow_model":
        model_path = ROOT / flow_params["model_path"]
        if not model_path.is_file():
            raise ValueError("No trained model available. Use Flow only.")
        flow_params["model_path"] = str(model_path)
        model_hash = digest_files([model_path])
    else:
        flow_params.pop("model_path", None)
    protocol = {"simulation": simulation_protocol(gui), "signals": config.get("signals"), "shield": config.get("shield"),
                "fixed_time": config.get("fixed_time"), "flow": {k:v for k,v in flow_params.items() if k != "model_path"},
                "max_pressure": config.get("max_pressure"), "rules": config.get("rules", {}), "delta_time": config["sim"]["delta_time"],
                "emission_rl": config.get("emission_rl", {}),
                "team_protocol": team, "orchestrator": config.get("orchestrator") if controller == "orchestrated" else None,
                "teleport_seconds": 60, "metric_schema": 2,
                "source_hash": digest_files([ROOT / p for p in ("flowmind/contracts.py", "flowmind/emission_rl.py", "flowmind/env.py", "flowmind/evaluation.py", "flowmind/kpis.py", "flowmind/agents/baselines.py", "flowmind/agents/flow.py", "flowmind/agents/rules.py",
                                                                    "flowmind/control/orchestrator.py", "flowmind/control/coordinator.py", "flowmind/control/shield.py",
                                                                    "flowmind/agents/emergency.py", "flowmind/agents/incident.py", "flowmind/agents/pedestrian.py", "flowmind/agents/bus.py")])}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + controller + "_" + uuid.uuid4().hex[:6]
    directory = ROOT / "runs" / run_id
    directory.mkdir(parents=True)
    meta = {"run_id": run_id, "controller": controller, "scenario": scenario,
            "situation": "trafic_fourni", "seconds": int(seconds), "seed": int(seed),
            "input_hash": digest_files([ROOT / net, ROOT / route]),
            "protocol_hash": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16],
            "model_hash": model_hash, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat()}
    if emission_manifest and (controller == "emission_model" or runtime_control):
        meta.update(emission_model_qualification=emission_manifest.get("qualification", "experimental"),
                    emission_action_mode=emission_manifest.get("action_mode", "phase"),
                    emission_reward_mode=emission_manifest.get("reward_mode", "weighted"),
                    emission_model_timesteps=emission_manifest.get("actual_timesteps"),
                    emission_model_directory=str(model_directory.resolve()),
                    emission_model_deterministic=emission_manifest.get("deterministic_inference", True),
                    emission_policy_seed=(int(seed) + int(emission_manifest.get("policy_seed_offset", 100000))
                                          if not emission_manifest.get("deterministic_inference", True) else None))
    atomic_json(directory / "metadata.json", meta | {"config": config, "protocol": protocol})
    config["evaluation_output"] = str(directory)
    config["gui_delay_ms"] = int(gui_delay_ms)
    recorder = Recorder(directory, meta)
    env = SumoCorridorEnv(config=config, use_gui=gui, seed=int(seed), begin_time=0 if scenario in TEAM_SCENARIOS else None)
    # Orchestrated runs add the per-agent figures of the last decision to every 1 s sample.
    agent_live = team_live({}, {}, [], 0) if controller == "orchestrated" else {}
    env.metrics_callback = lambda values: recorder.sample(values | agent_live)
    team_log = None
    status, error = "completed", None
    applied_revision = -1
    control_events = []
    try:
        states = env.reset()
        if controller == "emission_model":
            emission_policy = EmissionPolicy(scenario, config["rules"] | {k:config["signals"][k] for k in ("min_green","max_green")}, directory=model_directory, seed=seed)
        agents = {}
        for tls in env.tls_ids:
            if controller == "fixed_time":
                agents[tls] = FixedTime(config["fixed_time"])
            elif controller == "max_pressure":
                agents[tls] = MaxPressure(config.get("max_pressure", {}), env.get_phase_movements(tls))
            elif controller in ("rules", "emission_model"):
                agents[tls] = RuleBasedAgent(config.get("rules", {}) | {
                    "min_green": config["signals"]["min_green"],
                    "max_green": config["signals"]["max_green"],
                })
            elif controller == "orchestrated":
                # Same system as scripts/run.py --controller orchestrated (added by Adem)
                from scripts.run import make_orchestrator
                agents[tls] = make_orchestrator(config, env.get_phase_movements(tls))
            else:
                agents[tls] = FlowAgent(flow_params)
                if controller == "flow_model" and agents[tls].model is None:
                    raise ValueError("The model could not be loaded; run cancelled.")
        if controller == "orchestrated":
            from flowmind.control.coordinator import Coordinator
            from flowmind.control.shield import SafetyShield
            from flowmind.logger import DecisionLogger
            from scripts.run import warning_text
            order = [t for t in config.get("corridor", {}).get("order", []) if t in env.tls_ids] or list(env.tls_ids)
            coordinator = Coordinator(config["coordinator"], order)
            shield = SafetyShield(config["signals"], config.get("shield", {}))
            # the team's own log (decisions.csv / summary.csv), readable by scripts/explain.py
            team_log = DecisionLogger(directory / "team", latency_budget_ms=config["orchestrator"]["max_decision_ms"])
            colors = config.get("overlay", {}).get("colors", {})
            team_decisions, shield_changes = [], 0
        with (directory / "decisions.csv").open("w", newline="", encoding="utf-8-sig") as f:
            log = csv.DictWriter(f, fieldnames=["elapsed_s", "junction", "previous_phase", "requested_phase", "applied_phase", "reason"])
            log.writeheader()
            done = False
            while not done:
                if stop_event and stop_event.is_set():
                    status = "stopped"
                    break
                if runtime_control:
                    control = runtime_control.snapshot()
                    if control["revision"] != applied_revision:
                        previous_agents = agents
                        agents = {}
                        if control["method"] == "emission_model":
                            if emission_manifest and seed in emission_manifest["train_seeds"] + emission_manifest["validation_seeds"]:
                                raise ValueError("Seed de test distincte de l’entraînement/validation requise")
                            emission_policy = EmissionPolicy(scenario, control["rules"], directory=model_directory, seed=seed)
                        for tls in env.tls_ids:
                            if control["method"] == "rules":
                                agent = RuleBasedAgent(control["rules"])
                                old = previous_agents.get(tls)
                                if isinstance(old, RuleBasedAgent):
                                    agent.waiting_since = dict(old.waiting_since)
                                    agent.last_time = old.last_time
                                agents[tls] = agent
                            else:
                                agents[tls] = FixedTime(config["fixed_time"])
                        env.min_green = control["rules"]["min_green"]
                        env.max_green = control["rules"]["max_green"]
                        applied_revision = control["revision"]
                        recorder.metadata.update(controller=control["method"], control_revision=applied_revision)
                        control_events.append(control | {"elapsed_s": recorder.last.get("elapsed_s", 0)})
                        atomic_json(directory / "control_events.json", {"events": control_events})
                method = recorder.metadata["controller"]
                model_reasons = {}
                if method == "emission_model":
                    actions, model_reasons = emission_policy.act_all(states)
                elif method == "orchestrated":
                    now = next(iter(states.values())).time
                    events = coordinator.update(now, env.emergency_approaches(), team_decisions, states)
                    team_decisions, actions = [], {}
                    for tls, st in states.items():
                        d = agents[tls].decide(st, events.get(tls, []), shield.allowed_phases(st))
                        actions[tls] = d.final
                        model_reasons[tls] = f"{d.agent}: {d.explanation}"
                        shield_changes += bool(d.shield_note)
                        team_log.log(d, events.get(tls, []))
                        team_decisions.append(d)
                        env.show(tls, f"{tls} | {d.agent.upper()}: {d.explanation}", colors.get(d.agent, [128, 128, 128, 255]),
                                 warning_text(events.get(tls, [])))
                    agent_live = team_live(states, events, team_decisions, shield_changes)
                else:
                    actions = {tls: agents[tls].act(st, [], [True] * st.num_phases) for tls, st in states.items()}
                if runtime_control:
                    for tls, st in states.items():
                        if st.time_in_phase < env.min_green:
                            actions[tls] = st.phase
                previous = states
                states, _, _ = env.step(actions)
                # Continue an empty network to the same horizon for every trial.
                done = recorder.last.get("elapsed_s", 0) >= seconds
                for tls, action in actions.items():
                    log.writerow({"elapsed_s": recorder.last.get("elapsed_s", 0), "junction": tls,
                                  "previous_phase": previous[tls].phase, "requested_phase": action,
                                  "applied_phase": states[tls].phase, "reason": model_reasons.get(tls) or agents[tls].explain()})
                f.flush()
    except Exception as exc:
        status, error = "failed", str(exc)
    finally:
        env.close()
        recorder.close()
    result = meta | {"status": status, "error": error}
    if team and (directory / "tripinfo.xml").exists():
        try:
            result.update(team_kpis(directory / "tripinfo.xml"))
        except Exception as exc:
            result.update(team_kpis_error=str(exc))
    if team_log is not None:
        total = sum(team_log.control_steps.values()) or 1
        result["control_share"] = {a: round(100 * n / total, 1) for a, n in team_log.control_steps.most_common()}
        result["shield_changes"] = shield_changes
        result.update(team_log.stats())
        team_log.close({k: result[k] for k in ("team_arrived", "team_mean_waiting_s", "team_mean_trip_s", "ambulance_trip_s")
                        if k in result} | team_log.stats())
    if runtime_control:
        result.update(initial_controller=controller, control_changes=max(0, len(control_events)-1))
        if len(control_events) > 1:
            result["controller"] = "interactive"
    if (directory / "tripinfo.xml").exists() and recorder.last:
        try:
            result.update(compute_kpis(directory))
        except Exception as exc:
            result.update(status="failed", error=f"Metric computation: {exc}")
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "live.json", result | recorder.last)
    if result["status"] == "failed":
        raise RuntimeError(result["error"])
    return result


def load_results():
    rows = []
    for path in sorted((ROOT / "runs").glob("*/result.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            continue
    return rows


def latest_comparisons(rows):
    unique = {}
    for row in rows:
        if row["status"] == "completed":
            key = tuple(row[k] for k in ("controller", "scenario", "situation", "seconds", "seed", "input_hash", "protocol_hash", "model_hash"))
            unique[key] = row
    return compare_rows(list(unique.values()))


def export_results():
    rows = load_results()
    comparisons = latest_comparisons(rows)
    output = ROOT / "runs" / "reports"
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    write_csv(output / "metrics.csv", [{k:r.get(k) for k in fields} for r in rows])
    write_csv(output / "comparison.csv", comparisons)
    atomic_json(output / "comparison.json", {"comparisons": comparisons})
    if comparisons:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = {}
        for r in comparisons:
            key = (r["scenario"], r["seconds"], r["input_hash"], r["protocol_hash"], r["controller"], r["model_hash"])
            groups.setdefault(key, []).append(r)
        for key, group in groups.items():
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for ax, metric in zip(axes, ["mean_wait_s", "mean_queue_per_lane", "co2_kg"]):
                values = [r for r in group if r["metric"] == metric]
                if values:
                    ax.bar(["fixed_time"] + [r["controller"] for r in values], [values[0]["baseline_mean"]] + [r["controller_mean"] for r in values], color=["#64748b"] + ["#0891b2"] * len(values))
                    ax.set_title(values[0]["label"])
                    ax.set_ylabel(values[0]["unit"])
                    ax.tick_params(axis="x", rotation=20)
            fig.suptitle(f"{key[0]} · {key[1]} s · essais appariés")
            fig.tight_layout()
            fig.savefig(output / f"comparison_{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}_{key[5]}.png", dpi=140)
            plt.close(fig)
    return comparisons
