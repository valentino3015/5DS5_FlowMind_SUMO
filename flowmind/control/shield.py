"""Safety shield (the safety guard): checks every action before it reaches SUMO.

Owner: pedestrian agent owner (TBD).

TODO:
- build the allowed list given to act(): e.g. no switch before signals.min_green,
  force a switch after signals.max_green, no switch while pedestrians are still
  on a crossing (shield.ped_clearance_speed), serve pedestrians who waited
  more than shield.ped_max_wait
- if an action is unsafe, replace it and write why
"""
