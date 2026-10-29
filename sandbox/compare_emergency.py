"""Compare fixed_time vs fixed_time + EmergencyAgent on a scenario with ambulances.

Run from the repo root:  python -m sandbox.compare_emergency --seeds 10
"""
from __future__ import annotations

import argparse
import copy
import statistics as stats
import xml.etree.ElementTree as ET
from pathlib import Path

import sumolib
import yaml

from flowmind import env as env_module
from flowmind.agents.baselines import FixedTime
from flowmind.agents.emergency import EmergencyAgent
from flowmind.env import SumoCorridorEnv, _get_first_depart

ROOT = Path(__file__).resolve().parent.parent
SRC_ROU = next((ROOT / "scenarios").glob("*.rou.xml"))
NET = next((ROOT / "scenarios").glob("*.net.xml"))
OUT_ROU = Path(__file__).resolve().parent / "cologne3_amb.rou.xml"


def build_route_file(n_amb: int, first_offset: int = 100, spacing: int = 200) -> int:
    """Copy the scenario and add ambulances on routes that cross the most traffic lights."""
    net = sumolib.net.readNet(str(NET))
    tree = ET.parse(SRC_ROU)
    root = tree.getroot()
    vehicles = root.findall("vehicle")

    def tls_count(edges):
        n = 0
        for e in edges:
            try:
                if net.getEdge(e).getToNode().getType().startswith("traffic_light"):
                    n += 1
            except KeyError:
                return -1
        return n

    scored = []
    for v in vehicles:
        edges = v.find("route").get("edges").split()
        scored.append((tls_count(edges), len(edges), edges))
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: (-s[0], -s[1]))

    seen, picks = set(), []
    for n_tls, _, edges in scored:
        if tuple(edges) not in seen:
            seen.add(tuple(edges))
            picks.append((n_tls, edges))
        if len(picks) == n_amb:
            break

    first = _get_first_depart(SRC_ROU)
    vtype = ET.Element("vType", id="amb", vClass="emergency", length="6.5",
                       maxSpeed="25", accel="3.0", decel="5.0")
    ambs = []
    for i, (n_tls, edges) in enumerate(picks):
        v = ET.Element("vehicle", id=f"amb_{i}", type="amb",
                       depart=str(first + first_offset + i * spacing))
        ET.SubElement(v, "route", edges=" ".join(edges))
        ambs.append(v)
        print(f"  amb_{i}: {len(edges)} edges, crosses {n_tls} traffic light(s)")

    others = [c for c in root if c.tag != "vehicle"]
    allv = vehicles + ambs
    allv.sort(key=lambda v: float(v.get("depart")))
    for c in list(root):
        root.remove(c)
    for k in list(root.attrib):
        del root.attrib[k]
    for c in others:
        root.append(c)
    root.append(vtype)
    for v in allv:
        root.append(v)
    tree.write(OUT_ROU, encoding="utf-8", xml_declaration=True)
    return len(picks)


def run_one(controller: str, seed: int, seconds: int, base_config: dict) -> dict:
    config = copy.deepcopy(base_config)
    config.setdefault("sim", {})["seconds"] = seconds
    env = SumoCorridorEnv(config=config, use_gui=False, seed=seed, route_file=OUT_ROU)
    tc = env_module.traci
    states = env.reset()

    fixed = {t: FixedTime(config.get("fixed_time", {})) for t in env.tls_ids}
    emerg = {t: EmergencyAgent(config.get("emergency", {})) for t in env.tls_ids}

    first_seen, last_seen, stopped = {}, {}, {}
    takeover, halting, speeds = 0, [], []
    now, done = 0.0, False
    try:
        while not done:
            actions = {}
            for tls, s in states.items():
                allowed = [True] * s.num_phases
                if controller == "emergency" and emerg[tls].wants_control(s, []) is not None:
                    actions[tls] = emerg[tls].act(s, [], allowed)
                    takeover += 1
                else:
                    actions[tls] = fixed[tls].act(s, [], allowed)
            states, done, info = env.step(actions)
            halting.append(info["halting_vehicles"])
            speeds.append(info["mean_speed"])
            now = info["time"]
            for vid in tc.vehicle.getIDList():
                if vid.startswith("amb_"):
                    first_seen.setdefault(vid, now)
                    last_seen[vid] = now
                    if tc.vehicle.getSpeed(vid) < 0.1:
                        stopped[vid] = stopped.get(vid, 0) + env.delta_time
    finally:
        env.close()

    finished = [v for v in last_seen if last_seen[v] < now - env.delta_time]
    nan = float("nan")
    return {
        "amb_seen": len(first_seen),
        "amb_done": len(finished),
        "stop": stats.mean(stopped.get(v, 0) for v in finished) if finished else nan,
        "travel": stats.mean(last_seen[v] - first_seen[v] for v in finished) if finished else nan,
        "takeover": takeover,
        "halting": stats.mean(halting),
        "speed": stats.mean(speeds),
    }


def ms(values):
    vals = [v for v in values if v == v]
    if not vals:
        return "n/a", float("nan")
    sd = stats.stdev(vals) if len(vals) > 1 else 0.0
    return f"{stats.mean(vals):7.1f} +/- {sd:4.1f}", stats.mean(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seconds", type=int, default=1200)
    ap.add_argument("--ambulances", type=int, default=3)
    args = ap.parse_args()

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Building scenario with ambulances...")
    build_route_file(args.ambulances)

    results = {"fixed_time": [], "emergency": []}
    for i in range(args.seeds):
        seed = 100 + i
        for ctrl in results:
            r = run_one(ctrl, seed, args.seconds, config)
            results[ctrl].append(r)
            print(f"seed {seed} | {ctrl:10s} | amb {r['amb_done']}/{r['amb_seen']} done"
                  f" | stop {r['stop']:5.1f}s | travel {r['travel']:6.1f}s"
                  f" | takeover steps {r['takeover']}")

    print("\n=== Mean +/- std over seeds ===")
    print(f"{'metric':32s} | {'fixed_time':18s} | {'fixed + emergency':18s} | change")
    rows = [("ambulance stopped time (s)", "stop"), ("ambulance travel time (s)", "travel"),
            ("halting vehicles (all traffic)", "halting"), ("mean speed (m/s)", "speed")]
    for label, key in rows:
        a_txt, a = ms([r[key] for r in results["fixed_time"]])
        b_txt, b = ms([r[key] for r in results["emergency"]])
        change = f"{(b - a) / a * 100:+.1f}%" if a == a and b == b and a != 0 else "n/a"
        print(f"{label:32s} | {a_txt:18s} | {b_txt:18s} | {change}")


if __name__ == "__main__":
    main()
