"""The emission agent is part of the FlowMind system: on above a fixed CO2 threshold, off below another."""
from dataclasses import replace

import pytest
import yaml

from flowmind.agents.emission import EmissionAgent, junction_co2, network_model_for
from flowmind.contracts import LaneEmissions, make_example_state
from flowmind.env import ROOT


class FakeNetwork:
    def __init__(self, actions):
        self.time, self.actions = None, actions
        self.reasons = {k: f"model chose {v}" for k, v in actions.items()}


def at(time, co2):
    """Example junction at `time` whose incoming lanes emit `co2` mg/s in total."""
    return replace(make_example_state(), time=float(time),
                   lane_emissions=(LaneEmissions("in_0", co2 / 2, 0, 0, 0), LaneEmissions("in_1", co2 / 2, 0, 0, 0)))


def agent():
    return EmissionAgent({"window_s": 30, "activate_mg_s": 25000, "release_mg_s": 19000},
                         FakeNetwork({make_example_state().tls_id: 1}))


def config():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_junction_co2_sums_the_incoming_lanes():
    assert junction_co2(at(0, 300)) == pytest.approx(300)


def test_on_above_the_activation_threshold_off_below_the_release_threshold():
    a = agent()
    assert all(a.wants_control(at(t, 22000), []) is None for t in range(0, 60, 5))   # normal: Flow drives
    claims = [a.wants_control(at(t, 30000), []) for t in range(60, 120, 5)]
    assert claims[-1] is not None and "high CO2" in claims[-1].reason              # above 25 g/s: on
    assert all(a.wants_control(at(t, 22000), []) is not None for t in range(120, 180, 5))  # between: stays on
    assert all(a.wants_control(at(t, 15000), []) is None for t in range(180, 240, 5)[6:])  # below 19 g/s: off
    assert not a.active


def test_the_window_mean_decides_not_one_spike():
    a = agent()
    for t in range(0, 60, 5):
        a.wants_control(at(t, 20000), [])
    assert a.wants_control(at(60, 40000), []) is None  # (5 x 20 + 40) / 6 = 23.3 g/s < 25 g/s


def test_release_threshold_cannot_exceed_activation_threshold():
    with pytest.raises(ValueError):
        EmissionAgent({"activate_mg_s": 10000, "release_mg_s": 20000}, FakeNetwork({}))


def test_act_returns_the_shared_model_choice():
    a = agent()
    state = at(10, 30000)
    a.network.time = state.time
    a.wants_control(state, [])
    assert a.act(state, [], [True] * state.num_phases) == 1
    a.network.time = 5.0
    with pytest.raises(RuntimeError):
        a.act(state, [], [True] * state.num_phases)


def test_emission_agent_sits_just_above_flow():
    from scripts.run import make_orchestrator
    alone = make_orchestrator(config(), {})
    assert "emission" not in alone.agents and alone.priority[-1] == "flow"
    combined = make_orchestrator(config(), {}, FakeNetwork({}))
    assert {"emission", "flow"} <= set(combined.agents)
    assert combined.priority[-2:] == ["emission", "flow"]
    assert combined.priority[:-2] == alone.priority[:-1]


def test_model_only_on_scenarios_that_have_one_and_when_enabled():
    cfg = config()
    assert cfg["emission_agent"]["enabled"] is True
    assert network_model_for(cfg, "corridor_demo") is not None
    assert network_model_for(cfg, "corridor") is not None
    assert network_model_for(cfg, "grid10_demo") is None and network_model_for(cfg, "corridor5") is None
    cfg["emission_agent"]["enabled"] = False
    assert network_model_for(cfg, "corridor_demo") is None
