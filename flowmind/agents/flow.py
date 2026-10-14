"""Flow agent: handles normal traffic. Learns by itself (PPO reinforcement learning).

Owner: flow agent owner (TBD). Training script: scripts/train.py (same owner).

What to implement:
- wants_control: the flow agent is the default. It should always return a Claim
  with low urgency (e.g. 0.1), so it drives the junction when nobody else needs to.
- act: load the trained PPO model (path in config.yaml "flow: model_path"),
  turn the JunctionState into an observation (e.g. queue of each lane +
  one-hot of the current phase), and return model.predict(obs).
  If the chosen phase is not allowed, return self.keep_phase(state, allowed).
- explain: e.g. "PPO chose phase 0 (north-south queue 13)".

Simple example rule to start with, before the model exists:
    # give green to the phase whose lanes have the longest total queue
"""
from __future__ import annotations

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState


class FlowAgent(Specialist):
    name = "flow"

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        return None  # TODO(flow owner): return Claim(0.1, "normal traffic")

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        # TODO(flow owner): use the trained PPO model here.
        self.last_reason = "stub: keep current phase"
        return self.keep_phase(state, allowed)
