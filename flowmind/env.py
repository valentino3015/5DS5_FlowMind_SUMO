"""SUMO connection: the ONLY file that imports traci / libsumo.

Owner: Adem.

TODO (next phase):
- start SUMO with the scenario from scenarios/ (seed, gui from config.yaml)
- every sim.delta_time seconds, build one JunctionState per traffic light
- apply the chosen green phase, inserting yellow (signals.yellow) and
  all-red (signals.all_red) automatically
- give the phase -> lane movements of each junction (needed by MaxPressure)
"""
