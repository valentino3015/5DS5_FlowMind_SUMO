"""Bounded, server-owned facts for the simulation explanation agent."""
from __future__ import annotations

from collections import deque
import csv
import json
import math
from pathlib import Path


FINAL_UNITS = {
    "actual_seconds": "s", "departed": "véhicules", "arrived": "véhicules",
    "unfinished": "trajets", "active_at_end": "véhicules", "pending_at_end": "véhicules",
    "mean_wait_s": "s", "max_wait_s": "s", "mean_completed_duration_s": "s",
    "mean_queue_per_lane": "véhicules/voie", "max_queue_one_lane": "véhicules",
    "mean_halting_network": "véhicules", "mean_speed_kmh": "km/h",
    "co2_kg": "kg", "co2_kg_per_departed": "kg/véhicule parti", "nox_g": "g",
    "pmx_g": "g", "fuel_kg": "kg", "collision_vehicle_events": "événements véhicule",
    "teleport_events": "événements",
}
SAMPLE_UNITS = {
    "vehicles": "véhicules", "halting": "véhicules", "queue_mean_lane": "véhicules/voie",
    "queue_max_lane": "véhicules", "waiting_mean_active_s": "s", "speed_kmh": "km/h",
    "co2_g_s": "g/s", "nox_mg_s": "mg/s", "pmx_mg_s": "mg/s", "fuel_mg_s": "mg/s",
}
STATUS_LABELS = {"running": "en cours", "completed": "terminée", "stopped": "interrompue",
                 "failed": "échouée"}


def number(value):
    try:
        value = float(value)
        return round(value, 4) if math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def build_context(root: Path, run_id: str) -> dict:
    """Read complete CSV rows, keeping at most 120 one-second samples in memory.

    Final SUMO totals are read only from result.json; never integrate downsampled
    chart data or treat TraCI emission rates as official final totals.
    """
    runs = (Path(root) / "runs").resolve()
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Aucune simulation disponible")
    directory = (runs / run_id).resolve()
    if directory.parent != runs:
        raise ValueError("Identifiant de simulation invalide")
    meta = read_json(directory / "metadata.json")
    if not meta or meta.get("run_id") != run_id:
        raise ValueError("Simulation introuvable")
    result = read_json(directory / "result.json")
    status = result.get("status", meta.get("status", "unknown"))
    recent = deque(maxlen=120)
    count, methods, revisions = 0, set(), set()
    try:
        with (directory / "timeseries.csv").open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(line for line in source if line.endswith("\n")):
                elapsed = number(row.get("elapsed_s"))
                vehicles = number(row.get("vehicles"))
                if elapsed is None or vehicles is None:
                    continue
                if recent and elapsed <= recent[-1]["elapsed_s"]:
                    continue
                sample = {key: number(row.get(key)) for key in SAMPLE_UNITS}
                speed = number(row.get("speed_sum_m_s"))
                co2 = number(row.get("co2_mg_s"))
                sample["speed_kmh"] = round(3.6 * speed / vehicles, 4) if vehicles and speed is not None else None
                sample["co2_g_s"] = round(co2 / 1000, 4) if co2 is not None else None
                sample["elapsed_s"] = elapsed
                recent.append(sample)
                count += 1
                method = row.get("controller", meta.get("controller"))
                if method in {"fixed_time", "rules", "emission_model", "flow_model", "flow_heuristic", "max_pressure"}:
                    methods.add(method)
                revision = row.get("control_revision", "0")
                if len(revisions) < 1000:
                    revisions.add(revision)
    except FileNotFoundError:
        pass
    stats = {}
    for key, unit in SAMPLE_UNITS.items():
        values = [p[key] for p in recent if p[key] is not None]
        if values:
            stats[key] = {"unit": unit, "mean": round(sum(values) / len(values), 4),
                          "max": max(values), "first": values[0], "last": values[-1],
                          "samples": len(values)}
    final = {key: {"value": number(result.get(key)), "unit": unit}
             for key, unit in FINAL_UNITS.items() if number(result.get(key)) is not None}
    conversions = {}
    for source, target, unit in (("co2_kg", "co2_total_g", "g"),
                                 ("fuel_kg", "fuel_total_g", "g"),
                                 ("nox_g", "nox_total_mg", "mg"),
                                 ("pmx_g", "pmx_total_mg", "mg")):
        if source in final:
            conversions[target] = {"value": round(final[source]["value"] * 1000, 2),
                                   "unit": unit, "source": source, "factor": 1000}
    return {
        "simulation": {"run_id": run_id, "scenario": meta.get("scenario"),
                       "controller": result.get("controller", meta.get("controller")),
                       "status": status, "status_fr": STATUS_LABELS.get(status, "inconnu"),
                       "seed": meta.get("seed"), "planned_seconds": meta.get("seconds"),
                       "sample_count": count, "observed_until_s": recent[-1]["elapsed_s"] if recent else 0,
                       "methods_observed": sorted(methods),
                       "controls_changed": len(methods) > 1 or len(revisions) > 1},
        "recent_window": {"from_s": recent[0]["elapsed_s"] if recent else None,
                          "to_s": recent[-1]["elapsed_s"] if recent else None,
                          "sample_count": len(recent), "metrics": stats},
        "final_metrics": final,
        "conversions": conversions,
        "kpi_scope": "bilan partiel" if status in ("stopped", "failed", "running") else "bilan final",
        "comparison": None,
        "definitions": [
            "Fenêtre récente : au plus 120 mesures à une seconde, pas toute la simulation.",
            "waiting_mean_active_s : temps d'arrêt courant moyen des véhicules présents ; différent de mean_wait_s, attente cumulée par trajet incluant les trajets inachevés.",
            "mean_completed_duration_s ne concerne que les trajets terminés.",
            "Les débits d'émissions TraCI sont instantanés ; les totaux finaux sont issus de tripinfo SUMO, trajets inachevés inclus.",
            "CO₂ : gaz à effet de serre ; NOx et PMx : polluants atmosphériques ; fuel : quantité de carburant consommée.",
            "Les conversions sont calculées en Python avant l'arrondi d'affichage ; multiplier une valeur déjà arrondie peut donner un autre résultat.",
            "Vitesse récente : moyenne temporelle des vitesses moyennes instantanées ; mean_speed_kmh final pondère par les véhicules-secondes.",
            "Zéro est une mesure, une métrique absente est inconnue. Une simulation stopped/failed est partielle.",
            "Aucune référence comparative fournie : aucun gain chiffré ni classement de méthodes ne peut être conclu.",
            "Un changement de commande ne démontre pas une causalité ; le trafic précédent reste présent.",
        ],
    }
