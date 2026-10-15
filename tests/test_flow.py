"""Unit tests for the Flow agent.

Owner: Ahmed (flow agent owner). Run with: pytest tests/test_flow.py
"""
import numpy as np
import pytest

from flowmind.agents.flow import FlowAgent
from flowmind.contracts import Claim, make_example_state


class MockModel:
    """Mock RL model for testing without requiring a trained file or heavy dependencies."""

    def __init__(self, action: int, fail: bool = False):
        self.action = action
        self.fail = fail

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if self.fail:
            raise RuntimeError("Inference engine error")
        return self.action, None


def test_flow_wants_control_returns_claim():
    s = make_example_state()
    agent = FlowAgent({"urgency": 0.15})
    claim = agent.wants_control(s, [])
    assert isinstance(claim, Claim)
    assert claim.urgency == 0.15
    assert "traffic" in claim.reason


def test_flow_heuristic_picks_highest_queue():
    s = make_example_state()
    agent = FlowAgent({})
    all_allowed = [True] * s.num_phases
    # In make_example_state(), phase 0 serves North-South which has total queue = 10 + 3 = 13 cars
    action = agent.act(s, [], all_allowed)
    assert action == 0
    assert "heuristic" in agent.explain()
    assert "highest queue" in agent.explain()


def test_flow_heuristic_respects_shield_allowed_mask():
    s = make_example_state()
    agent = FlowAgent({})
    # Block phase 0; current phase in make_example_state is 2
    allowed = [False, True, True, True]
    action = agent.act(s, [], allowed)
    assert action != 0
    assert allowed[action] is True


def test_encode_state_shape_and_range():
    s = make_example_state()
    agent = FlowAgent({"max_queue": 25.0, "max_wait": 120.0})
    obs = agent.encode_state(s)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.ndim == 1
    # Number of features:
    # 8 incoming lanes * 3 + 4 outgoing lanes * 1 + 4 phases one-hot + 1 time in phase = 33
    expected_dim = len(s.lanes) * 3 + len(s.out_lanes) * 1 + s.num_phases + 1
    assert obs.shape[0] == expected_dim

    # All normalized values must be between 0.0 and 1.0
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)
    assert not np.any(np.isnan(obs))

    # Check phase one-hot encoding
    one_hot_start = len(s.lanes) * 3 + len(s.out_lanes) * 1
    phase_slice = obs[one_hot_start : one_hot_start + s.num_phases]
    assert np.sum(phase_slice) == 1.0
    assert phase_slice[s.phase] == 1.0


def test_flow_act_with_mock_model_allowed():
    s = make_example_state()
    mock_model = MockModel(action=1)
    agent = FlowAgent({}, model=mock_model)
    all_allowed = [True] * s.num_phases

    action = agent.act(s, [], all_allowed)
    assert action == 1
    assert "PPO model chose phase 1" in agent.explain()


def test_flow_act_with_mock_model_blocked_by_shield():
    s = make_example_state()
    mock_model = MockModel(action=1)
    agent = FlowAgent({}, model=mock_model)
    # Block phase 1
    allowed = [True, False, True, True]

    action = agent.act(s, [], allowed)
    assert action != 1
    assert allowed[action] is True
    assert "shield blocked" in agent.explain()


def test_flow_act_with_failing_model_falls_back():
    s = make_example_state()
    failing_model = MockModel(action=0, fail=True)
    agent = FlowAgent({}, model=failing_model)
    all_allowed = [True] * s.num_phases

    # Should not raise exception, but cleanly fall back to heuristic
    action = agent.act(s, [], all_allowed)
    assert action == 0
    assert "heuristic" in agent.explain()
