"""Emission agent: Manoubi's emission RL model, switched on when a junction's emissions are high.

Owner: Manoubi.

Part of the FlowMind system (all agents) on the scenarios that have an
emission model (runs/models/emission_ppo_corridor: corridor and corridor_demo).
Flow drives normal traffic. When the CO2 of a junction's incoming lanes,
averaged over `window_s`, goes above the fixed `activate_mg_s` threshold, the
emission agent claims that junction; it ranks above Flow, so Flow stops
driving it. When the CO2 falls below the fixed `release_mg_s` threshold, the
emission agent stops claiming and Flow drives the junction again. Emergency,
incident, pedestrian and bus agents keep their priority, and the safety
shield still checks every phase. Thresholds: config.yaml, section emission_agent.

The model decides for the whole network at once, the orchestrator junction by
junction. One NetworkEmissionModel is therefore shared by the emission agents
of every junction: the simulation loop calls update(states) once per decision
step, before the orchestrators decide, and each agent reads its junction.
"""
from __future__ import annotations

from collections import deque
import json

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Event, JunctionState
from flowmind.emission_rl import EmissionPolicy, model_directory, serves


class NetworkEmissionModel:
    """The scenario's emission model, run once per step for every junction."""

    def __init__(self, scenario: str, params: dict, directory=None, seed: int | None = None):
        directory = model_directory(scenario) if directory is None else directory
        self.policy = EmissionPolicy(scenario, params, directory=directory, seed=seed)
        self.manifest = self.policy.manifest
        self.time = None
        self.actions: dict[str, int] = {}
        self.reasons: dict[str, str] = {}

    def update(self, states: dict[str, JunctionState]) -> None:
        """Model choices for this step. Runs every step, so the model's safeguards
        (starvation memory, min/max green) stay current while other agents drive."""
        self.actions, self.reasons = self.policy.act_all(states)
        self.time = next(iter(states.values())).time


def network_model_for(config: dict, scenario: str, seed: int | None = None) -> NetworkEmissionModel | None:
    """The shared model when the emission agent is on and the scenario has a model, else None (Flow only)."""
    if not config.get("emission_agent", {}).get("enabled", False):
        return None
    directory = model_directory(scenario)
    if not (directory / "best.zip").is_file() or not (directory / "manifest.json").is_file():
        return None
    if not serves(json.loads((directory / "manifest.json").read_text(encoding="utf-8")), scenario):
        return None
    params = config["rules"] | {k: config["signals"][k] for k in ("min_green", "max_green")}
    return NetworkEmissionModel(scenario, params, directory=directory, seed=seed)


def junction_co2(state: JunctionState) -> float:
    """CO2 of the junction's incoming lanes now (mg/s)."""
    return sum(lane.co2 for lane in state.lane_emissions)


class EmissionAgent(Specialist):
    """Specialist for high emissions: lowers CO2, NOx and fuel at that junction."""

    name = "emission"

    def __init__(self, params: dict | None, network: NetworkEmissionModel):
        super().__init__(params)
        self.network = network
        self.urgency = float(self.params.get("urgency", 0.1))
        self.window_s = float(self.params.get("window_s", 30))
        self.activate_mg_s = float(self.params.get("activate_mg_s", 19000))
        self.release_mg_s = float(self.params.get("release_mg_s", 15000))
        if self.release_mg_s > self.activate_mg_s:
            raise ValueError("emission_agent: release_mg_s must not exceed activate_mg_s")
        self.recent: deque[tuple[float, float]] = deque()  # (time, CO2 mg/s) over the window
        self.active = False
        self.level = 0.0

    def observe(self, state: JunctionState) -> float:
        """Record this step's CO2; return its mean over the window (mg/s)."""
        self.recent.append((state.time, junction_co2(state)))
        while self.recent[0][0] <= state.time - self.window_s:
            self.recent.popleft()
        return sum(c for _, c in self.recent) / len(self.recent)

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        self.level = self.observe(state)
        if self.level > self.activate_mg_s:
            self.active = True
        elif self.level < self.release_mg_s:
            self.active = False
        if not self.active:
            return None
        return Claim(urgency=self.urgency,
                     reason=f"high CO2: {self.level / 1000:.1f} g/s (on above {self.activate_mg_s / 1000:.1f}, "
                            f"off below {self.release_mg_s / 1000:.1f})")

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        if self.network.time != state.time or state.tls_id not in self.network.actions:
            raise RuntimeError("NetworkEmissionModel.update(states) must run before the orchestrators decide")
        self.last_reason = f"{self.network.reasons[state.tls_id]} (CO2 {self.level / 1000:.1f} g/s)"
        return int(self.network.actions[state.tls_id])
