"""Metrics from one-second samples and SUMO trip information."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import statistics
import time
import xml.etree.ElementTree as ET


def atomic_json(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    # On Windows, a reader/antivirus may briefly hold the destination open.
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.01)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute_kpis(directory: Path) -> dict:
    """Waiting includes unfinished trips; duration only includes arrivals.

    Use SUMO's trip emission totals, including unfinished trips. Instantaneous
    TraCI rates remain in the time series for inspection, not final integration:
    vehicle insertion/removal boundaries otherwise introduce discrepancies.
    Queue means cover all non-internal network lanes, including empty lanes.
    """
    with (directory / "timeseries.csv").open(encoding="utf-8-sig", newline="") as f:
        samples = [{k: float(v) for k, v in row.items() if k != "controller"} for row in csv.DictReader(f)]
    trips = ET.parse(directory / "tripinfo.xml").getroot().findall("tripinfo")
    finished = [t for t in trips if float(t.get("arrival", "-1")) >= 0]
    n = len(samples)
    vehicle_seconds = sum(s["vehicles"] for s in samples)
    waits = [float(t.get("waitingTime", "0")) for t in trips]
    emissions = [t.find("emissions") for t in trips]
    if trips and any(e is None for e in emissions):
        raise ValueError("Sorties d’émissions absentes pour certains véhicules")
    def emission_total(name, divisor):
        return sum(float(e.get(name, "0")) for e in emissions) / divisor
    co2 = emission_total("CO2_abs", 1_000_000)
    departed = int(sum(s["departed"] for s in samples))
    return {
        "actual_seconds": samples[-1]["elapsed_s"] if samples else 0,
        "departed": departed,
        "arrived": int(sum(s["arrived"] for s in samples)),
        "unfinished": len(trips) - len(finished),
        "active_at_end": int(samples[-1]["vehicles"]) if samples else 0,
        "pending_at_end": int(samples[-1]["pending"]) if samples else 0,
        "mean_wait_s": sum(waits) / len(waits) if waits else None,
        "max_wait_s": max(waits) if waits else None,
        "mean_completed_duration_s": sum(float(t.get("duration", "0")) for t in finished) / len(finished) if finished else None,
        "mean_queue_per_lane": sum(s["queue_mean_lane"] for s in samples) / n if n else None,
        "max_queue_one_lane": max((s["queue_max_lane"] for s in samples), default=0),
        "mean_halting_network": sum(s["halting"] for s in samples) / n if n else None,
        "mean_speed_kmh": 3.6 * sum(s["speed_sum_m_s"] for s in samples) / vehicle_seconds if vehicle_seconds else None,
        "co2_kg": co2,
        "co2_kg_per_departed": co2 / departed if departed else None,
        "nox_g": emission_total("NOx_abs", 1000),
        "pmx_g": emission_total("PMx_abs", 1000),
        "fuel_kg": emission_total("fuel_abs", 1_000_000),
        "collision_vehicle_events": int(sum(s["collisions"] for s in samples)),
        "teleport_events": int(sum(s["teleports"] for s in samples)),
    }


COMPARE_METRICS = {
    "mean_wait_s": ("Attente moyenne observée", "s", False),
    "mean_completed_duration_s": ("Durée des trajets terminés", "s", False),
    "mean_queue_per_lane": ("File moyenne par voie", "véhicules", False),
    "co2_kg": ("CO₂ total sur la fenêtre", "kg", False),
    "co2_kg_per_departed": ("CO₂ par véhicule parti", "kg", False),
    "nox_g": ("NOx totaux sur la fenêtre", "g", False),
    "pmx_g": ("PM totales sur la fenêtre", "g", False),
    "fuel_kg": ("Carburant consommé sur la fenêtre", "kg", False),
    "arrived": ("Véhicules arrivés", "véhicules", True),
}


def compare_rows(rows: list[dict]) -> list[dict]:
    """Pair on identical conditions and seed; aggregate paired trials only."""
    keys = ("scenario", "situation", "seconds", "seed", "input_hash", "protocol_hash")
    valid = [r for r in rows if r["status"] == "completed" and r["controller"] != "interactive"]
    baselines = {tuple(r[k] for k in keys): r for r in valid if r["controller"] == "fixed_time"}
    groups = {}
    group_keys = ("scenario", "situation", "seconds", "input_hash", "protocol_hash", "controller", "model_hash")
    for row in valid:
        if row["controller"] == "fixed_time":
            continue
        baseline = baselines.get(tuple(row[k] for k in keys))
        if baseline is not None:
            groups.setdefault(tuple(row[k] for k in group_keys), []).append((baseline, row))
    result = []
    for group, pairs in groups.items():
        for metric, (label, unit, higher_better) in COMPARE_METRICS.items():
            measured = [(b[metric], r[metric]) for b, r in pairs if b.get(metric) is not None and r.get(metric) is not None]
            if not measured:
                continue
            base = statistics.mean(b for b, _ in measured)
            candidate = statistics.mean(r for _, r in measured)
            change = 100 * (candidate - base) / base if base else None
            improvement = change if higher_better else (-change if change is not None else None)
            deltas = [r-b for b, r in measured]
            result.append(dict(zip(group_keys, group)) | {
                "metric": metric, "label": label, "unit": unit,
                "paired_seeds": len(measured), "baseline_mean": base,
                "controller_mean": candidate, "change_pct": change,
                "improvement_pct": improvement,
                "paired_delta_std": statistics.stdev(deltas) if len(deltas) > 1 else None,
            })
    return result
