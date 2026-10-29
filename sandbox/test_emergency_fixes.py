from dataclasses import replace

import yaml

from flowmind.agents.emergency import EmergencyAgent
from flowmind.contracts import Event, SpecialVehicle, make_example_state

PARAMS = yaml.safe_load(open("config.yaml", encoding="utf-8"))["emergency"]
base = make_example_state()  # tls "J1", time 1200, phase 2 green
T = base.time


def amb(speed=12.0, dist=150.0, lane="north_in_0"):
    return SpecialVehicle("amb_1", "emergency", lane, distance=dist, speed=speed, delay=None)


def ok(name):
    print("OK  ", name)


# 1. Le hold marche avec le VRAI pas de 5 s, puis se termine
a = EmergencyAgent(PARAMS)
a.wants_control(replace(base, special_vehicles=(amb(),)), [])
claim = a.wants_control(replace(base, time=T + 5), [])
assert claim and claim.urgency == PARAMS["urgency_hold"], claim
assert a.wants_control(replace(base, time=T + 10), []) is None
ok("1: hold a +5 s, fini a +10 s")

# 2. Warning sans lane_id -> pas de claim
a = EmergencyAgent(PARAMS)
ev = Event("emergency", "J1", expires_at=T + 60, data={"eta": 18.0})
assert a.wants_control(base, [ev]) is None
ok("2: warning sans lane_id -> None")

# 3. Warning avec une voie qui n'est pas dans ce carrefour -> None
ev = Event("emergency", "J1", expires_at=T + 60, data={"eta": 18.0, "lane_id": "inconnue_0"})
assert a.wants_control(base, [ev]) is None
ok("3: voie inconnue -> None")

# 4. Warning destine a un autre carrefour -> None
ev = Event("emergency", "J2", expires_at=T + 60, data={"eta": 18.0, "lane_id": "north_in_1"})
assert a.wants_control(base, [ev]) is None
ok("4: autre carrefour -> None")

# 5. Warning valide -> claim et phase 1 (north_in_1)
ev = Event("emergency", "J1", expires_at=T + 60, data={"eta": 18.0, "lane_id": "north_in_1"})
claim = a.wants_control(base, [ev])
assert claim and claim.urgency == PARAMS["urgency_warning"]
assert a.act(base, [ev], [True] * 4) == 1
ok("5: warning valide -> phase 1")

# 6. Les vehicules disparus sont oublies
a = EmergencyAgent(PARAMS)
a.wants_control(replace(base, special_vehicles=(amb(),)), [])
a.wants_control(replace(base, time=T + 20), [])
assert not a._last_moved and not a._ignored and not a._stall and not a._skip
ok("6: memoire purgee")

# 7. Ambulance immobile trop longtemps -> l'agent abandonne le controle
a = EmergencyAgent(PARAMS)
claim = None
for k in range(0, 120, 5):
    claim = a.wants_control(replace(base, time=T + k, special_vehicles=(amb(speed=0.0, dist=19.0),)), [])
assert claim is None, claim
ok("7: abandon apres max_stall_s")

print("Tous les tests passent.")
