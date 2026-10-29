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
stuck = {}


def report(tls, s, v, asked):
    print(f"\n=== {v.vehicle_id} bloquee sur {tls} a t={s.time:.0f} ===")
    print("lane:", v.lane_id, "| distance a la ligne:", round(v.distance, 1), "m")
    print("phases qui servent cette voie:", [p for p, l in enumerate(s.phase_lanes) if v.lane_id in l])
    print("phase actuelle:", s.phase, "| demandee:", asked, "| depuis:", s.time_in_phase, "s")
    print("prochain feu (id, index, distance, signal G/g=vert r=rouge):", tc.vehicle.getNextTLS(v.vehicle_id))
    print("vehicule devant (id, ecart):", tc.vehicle.getLeader(v.vehicle_id, 200))
    amb_pos = tc.vehicle.getLanePosition(v.vehicle_id)
    print("vehicules sur la voie, devant l'ambulance (id, position, vitesse):")
    for vid in tc.lane.getLastStepVehicleIDs(v.lane_id):
        pos = tc.vehicle.getLanePosition(vid)
        if pos > amb_pos:
            print("  ", vid, round(pos), round(tc.vehicle.getSpeed(vid), 1))
    print("voies de sortie de la phase actuelle (occupation 0..1):")
    for lane in s.phase_out_lanes[s.phase]:
        print("  ", lane, round(tc.lane.getLastStepOccupancy(lane), 2))
    box = sum(tc.lane.getLastStepHaltingNumber(l) for l in tc.lane.getIDList() if l.startswith(":"))
    print("vehicules a l'arret dans les carrefours (boites):", box)


try:
    done = False
    while not done:
        actions = {}
        for tls, s in states.items():
            allowed = [True] * s.num_phases
            if emerg[tls].wants_control(s, []) is not None:
                a = emerg[tls].act(s, [], allowed)
                actions[tls] = a
                for v in s.special_vehicles:
                    if v.kind == "emergency" and v.speed < 0.1 and v.distance > 3:
                        stuck[v.vehicle_id] = stuck.get(v.vehicle_id, 0) + 1
                        if stuck[v.vehicle_id] == 6:   # 6 pas x 5 s = 30 s
                            report(tls, s, v, a)
                            raise SystemExit
            else:
                actions[tls] = fixed[tls].act(s, [], allowed)
        states, done, info = env.step(actions)
    print("Aucune ambulance bloquee.")
finally:
    env.close()
