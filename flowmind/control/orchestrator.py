"""Orchestrator (the manager): picks which agent controls each junction.

Owner: Adem.

TODO (next phase):
- ask every agent wants_control(state, events)
- pick one using orchestrator.priority from config.yaml
  (emergency, incident, pedestrian, bus, flow), keeping the choice at least
  orchestrator.min_hold seconds to avoid flickering
- call act(), pass the action through the shield, log explain()
"""
