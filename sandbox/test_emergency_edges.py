from dataclasses import replace

from flowmind.agents.emergency import EmergencyAgent
from flowmind.contracts import Event, SpecialVehicle, make_example_state

PARAMS = {"detect_distance": 200, "preempt_eta_s": 20, "warning_eta_s": 30,
          "hold_after_pass_s": 5, "min_speed_mps": 3}
base = make_example_state()  # phase 2 green; phases: 0=N-S, 1=N-S left, 2=E-W, 3=E-W left
ALL = [True] * base.num_phases


def amb(vid, lane, dist, speed, kind="emergency"):
    return SpecialVehicle(vid, kind, lane, distance=dist, speed=speed, delay=None)


def ok(name):
    print("OK  ", name)


# A. Deux ambulances: on sert celle qui arrive le plus tot (ETA), pas la plus proche
a = EmergencyAgent(PARAMS)
s = replace(base, special_vehicles=(amb("amb_a", "east_in_0", 80, 3), amb("amb_b", "north_in_0", 150, 15)))
claim = a.wants_control(s, [])
assert claim and "amb_b" in claim.reason, claim
assert a.act(s, [], ALL) == 0
ok("A: deux ambulances -> plus petit ETA")

# B. Le shield bloque la phase voulue: on garde la phase actuelle
a = EmergencyAgent(PARAMS)
s = replace(base, special_vehicles=(amb("amb_1", "north_in_0", 150, 12),))
a.wants_control(s, [])
assert a.act(s, [], [False, True, True, True]) == 2
ok("B: shield bloque -> keep_phase")

# C. Ambulance trop loin (> detect_distance): pas de claim
a = EmergencyAgent(PARAMS)
s = replace(base, special_vehicles=(amb("amb_1", "north_in_0", 250, 12),))
assert a.wants_control(s, []) is None
ok("C: trop loin -> None")

# D. Warning du coordinateur avec lane_id (voie de gauche nord -> phase 1)
a = EmergencyAgent(PARAMS)
ev = Event("emergency", "J1", expires_at=base.time + 60, data={"eta": 18.0, "lane_id": "north_in_1"})
claim = a.wants_control(base, [ev])
assert claim and claim.urgency == 0.7, claim
assert a.act(base, [ev], ALL) == 1
ok("D: warning avec lane_id -> phase 1")

# E. Warning sans lane_id: claim quand meme, on garde la phase actuelle
a = EmergencyAgent(PARAMS)
ev = Event("emergency", "J1", expires_at=base.time + 60, data={"eta": 18.0})
assert a.wants_control(base, [ev]) is None
assert a.act(base, [ev], ALL) == base.phase
ok("E: warning sans lane_id -> pas de claim")

# F. Warning expire: ignore
a = EmergencyAgent(PARAMS)
ev = Event("emergency", "J1", expires_at=base.time - 1, data={"eta": 18.0, "lane_id": "north_in_1"})
assert a.wants_control(base, [ev]) is None
ok("F: warning expire -> None")

# G. Apres le hold (5 s), l'agent relache le controle
a = EmergencyAgent(PARAMS)
a.wants_control(replace(base, special_vehicles=(amb("amb_1", "north_in_0", 150, 12),)), [])
assert a.wants_control(replace(base, time=base.time + 6), []) is None
ok("G: hold termine -> None")

# H. Un bus n'est pas une urgence
a = EmergencyAgent(PARAMS)
s = replace(base, special_vehicles=(amb("bus_1", "north_in_0", 50, 8, kind="bus"),))
assert a.wants_control(s, []) is None
ok("H: bus ignore -> None")

print("Tous les cas limites passent.")
