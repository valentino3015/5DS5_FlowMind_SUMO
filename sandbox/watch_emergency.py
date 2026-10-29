"""Watch the emergency agent in SUMO-GUI.

python -m sandbox.watch_emergency                      # fixed_time + emergency agent
python -m sandbox.watch_emergency --agent fixed_time   # same scenario, fixed timers only
"""
from __future__ import annotations

import argparse
import time

import yaml

from sandbox.compare_emergency import OUT_ROU, ROOT, build_route_file
from flowmind import env as env_module
from flowmind.agents.baselines import FixedTime
from flowmind.agents.emergency import EmergencyAgent
from flowmind.env import SumoCorridorEnv

ap = argparse.ArgumentParser()
ap.add_argument("--agent", choices=["emergency", "fixed_time"], default="emergency")
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--seconds", type=int, default=1200)
ap.add_argument("--delay", type=float, default=0.25, help="pause between decision steps (s)")
args = ap.parse_args()

config = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
config["sim"]["seconds"] = args.seconds
if not OUT_ROU.exists():
    build_route_file(3)

env = SumoCorridorEnv(config=config, use_gui=True, seed=args.seed, route_file=OUT_ROU)
tc = env_module.traci
states = env.reset()
fixed = {t: FixedTime(config.get("fixed_time", {})) for t in env.tls_ids}
emerg = {t: EmergencyAgent(config.get("emergency", {})) for t in env.tls_ids}
VIEW = "View #0"
tracked = None

try:
    try:
        tc.gui.setSchema(VIEW, "real world")
    except Exception:
        pass
    done = False
    while not done:
        actions = {}
        for tls, s in states.items():
            allowed = [True] * s.num_phases
            if args.agent == "emergency" and emerg[tls].wants_control(s, []) is not None:
                actions[tls] = emerg[tls].act(s, [], allowed)
                print(f"t={s.time:.0f} {tls[:12]:12s} EMERGENCY AGENT -> phase {actions[tls]} | {emerg[tls].explain()}")
            else:
                actions[tls] = fixed[tls].act(s, [], allowed)
        states, done, info = env.step(actions)

        ids = tc.vehicle.getIDList()
        for vid in ids:
            if vid.startswith("amb_"):
                try:
                    tc.vehicle.setColor(vid, (255, 0, 0, 255))
                    tc.vehicle.setShapeClass(vid, "emergency")
                except Exception:
                    pass
        try:
            if tracked is not None and tracked not in ids:
                tc.gui.trackVehicle(VIEW, "")
                tracked = None
            if tracked is None:
                nxt = next((v for v in ids if v.startswith("amb_")), None)
                if nxt:
                    tc.gui.trackVehicle(VIEW, nxt)
                    tc.gui.setZoom(VIEW, 700)
                    tracked = nxt
        except Exception:
            pass
        time.sleep(args.delay)
    print("Simulation finished.")
finally:
    env.close()