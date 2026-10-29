from dataclasses import replace

from flowmind.agents.emergency import EmergencyAgent
from flowmind.contracts import SpecialVehicle, make_example_state

agent = EmergencyAgent({
    "detect_distance": 200,
    "preempt_eta_s": 20,
    "warning_eta_s": 30,
    "hold_after_pass_s": 5,
    "min_speed_mps": 3,
})
base = make_example_state()  # phase 2 is green
allowed = [True] * base.num_phases

print("Case 1: no ambulance")
print(" claim:", agent.wants_control(base, []))

print("Case 2: ambulance on north_in_0, 150 m away")
amb = SpecialVehicle("amb_1", "emergency", "north_in_0", distance=150.0, speed=12.0, delay=None)
s = replace(base, special_vehicles=(amb,))
print(" claim:", agent.wants_control(s, []))
print(" phase:", agent.act(s, [], allowed), "|", agent.last_reason)

print("Case 3: ambulance just left, 3 s later")
s2 = replace(base, time=base.time + 3, special_vehicles=())
print(" claim:", agent.wants_control(s2, []))
print(" phase:", agent.act(s2, [], allowed), "|", agent.last_reason)
