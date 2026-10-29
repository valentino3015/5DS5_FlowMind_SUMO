import yaml

from sandbox.compare_emergency import OUT_ROU, ROOT
from flowmind import env as env_module
from flowmind.agents.baselines import FixedTime
from flowmind.agents.emergency import EmergencyAgent
from flowmind.env import SumoCorridorEnv

config = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
config["sim"]["seconds"] = 1200
env = SumoCorridorEnv(config=config, use_gui=False, seed=100, route_file=OUT_ROU)
tc = env_module.traci
states = env.reset()
fixed = {t: FixedTime(config.get("fixed_time", {})) for t in env.tls_ids}
emerg = {t: EmergencyAgent(config.get("emergency", {})) for t in env.tls_ids}

try:
    done = False
    while not done:
        actions = {}
        for tls, s in states.items():
            allowed = [True] * s.num_phases
            claim = emerg[tls].wants_control(s, [])
            if claim:
                a = emerg[tls].act(s, [], allowed)
                actions[tls] = a
                spd = [f"{v.vehicle_id}:{v.speed:.1f}m/s" for v in s.special_vehicles if v.kind == "emergency"]
                print(f"t={s.time:5.0f} {tls[:10]:10s} phase={s.phase} t_in={s.time_in_phase:3.0f} "
                      f"trans={int(s.in_transition)} asks={a} {spd} | {claim.reason}")
            else:
                actions[tls] = fixed[tls].act(s, [], allowed)
        states, done, info = env.step(actions)

    print("\n--- Ambulances still in the network at the end ---")
    for vid in tc.vehicle.getIDList():
        if vid.startswith("amb_"):
            print(vid, "lane:", tc.vehicle.getLaneID(vid), "pos:", round(tc.vehicle.getLanePosition(vid)),
                  "speed:", round(tc.vehicle.getSpeed(vid), 1))
finally:
    env.close()
