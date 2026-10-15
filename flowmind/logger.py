"""Logger: saves every decision, so we can show WHY the lights did what they did.

Owner: Adem.

Writes into one folder per run (runs/<date>_<controller>_seed<N>/):
- decisions.csv: one row per junction per step (who had control, why, which
  phase it asked for, what the shield changed, the coordinator's warnings)
- summary.csv: the totals at the end of the run
"""
from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

COLUMNS = ["time", "tls_id", "agent", "urgency", "reason", "proposed", "final",
           "shield_note", "explanation", "events"]


class DecisionLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.file = open(self.run_dir / "decisions.csv", "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(COLUMNS)
        self.control_steps = Counter()  # how many steps each agent was in control

    def log(self, decision, events: list) -> None:
        d = decision
        warnings = "; ".join(f"{e.kind} {e.data.get('vehicle_id', '')} eta {e.data.get('eta', 0):.0f}s"
                             for e in events)
        self.writer.writerow([d.time, d.tls_id, d.agent, round(d.urgency, 2), d.reason, d.proposed,
                              d.final, d.shield_note, d.explanation, warnings])
        self.control_steps[d.agent] += 1

    def close(self, summary: dict) -> None:
        """Write summary.csv (summary + share of steps per agent) and close the files."""
        self.file.close()
        total = sum(self.control_steps.values()) or 1
        rows = {}
        for key, value in summary.items():
            if isinstance(value, dict):  # e.g. {"amb_1": 93.0} -> emergency_travel_s_amb_1
                rows.update({f"{key}_{k}": v for k, v in value.items()})
            else:
                rows[key] = value
        for agent, steps in sorted(self.control_steps.items()):
            rows[f"control_share_{agent}"] = f"{100 * steps / total:.1f}%"
        with open(self.run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerows(rows.items())


def summarize_tripinfo(path: Path) -> dict:
    """Totals from SUMO's tripinfo file (one entry per vehicle that finished its trip).

    mean_waiting_per_vehicle_s: average time spent standing still (SUMO "waitingTime").
    emergency_travel_s: trip duration of each emergency vehicle (vType or id with "emergency"/"amb").
    """
    path = Path(path)
    if not path.is_file():
        return {}
    trips = [t.attrib for t in ET.parse(path).getroot().iter("tripinfo")]
    n = max(len(trips), 1)
    emergency = {t["id"]: float(t["duration"]) for t in trips
                 if "emergency" in t.get("vType", "").lower() or "amb" in t["id"].lower()}
    return {
        "arrived_vehicles": len(trips),
        "mean_waiting_per_vehicle_s": round(sum(float(t["waitingTime"]) for t in trips) / n, 2),
        "mean_trip_duration_s": round(sum(float(t["duration"]) for t in trips) / n, 2),
        "emergency_travel_s": emergency,
    }
