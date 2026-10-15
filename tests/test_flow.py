"""Unit tests for the Flow agent (V2 — includes spillback-aware logic).

Owner: Ahmed (flow agent owner). Run with: pytest tests/test_flow.py
"""
import numpy as np
import pytest

from flowmind.agents.flow import FlowAgent
from flowmind.contracts import Claim, LaneState, make_example_state


class MockModel:
    """Mock RL model for testing without requiring a trained file."""

    def __init__(self, action: int, fail: bool = False):
        self.action = action
        self.fail = fail

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if self.fail:
            raise RuntimeError("Inference engine error")
        return self.action, None


# ------------------------------------------------------------------ #
#  Basic interface contract                                            #
# ------------------------------------------------------------------ #

def test_flow_wants_control_returns_claim():
    s = make_example_state()
    agent = FlowAgent({"urgency": 0.15})
    claim = agent.wants_control(s, [])
    assert isinstance(claim, Claim)
    assert claim.urgency == 0.15
    assert "traffic" in claim.reason


# ------------------------------------------------------------------ #
#  Heuristic logic                                                     #
# ------------------------------------------------------------------ #

def test_flow_heuristic_picks_highest_queue():
    s = make_example_state()
    agent = FlowAgent({})
    all_allowed = [True] * s.num_phases
    # make_example_state phase 0 has highest queue
    action = agent.act(s, [], all_allowed)
    assert action == 0
    assert "heuristic" in agent.explain()
    assert "highest queue" in agent.explain()


def test_flow_heuristic_respects_shield_allowed_mask():
    s = make_example_state()
    agent = FlowAgent({})
    # Block phase 0
    allowed = [False, True, True, True]
    action = agent.act(s, [], allowed)
    assert action != 0
    assert allowed[action] is True


def test_flow_heuristic_skips_spillback_blocked_phase():
    """Heuristic should skip phases whose downstream lanes are jammed."""
    s = make_example_state()
    # Simulate spillback: give phase 0 outgoing lanes 100% occupancy
    # The example state phase_lanes[0] includes two incoming lanes
    # We mock phase_out_lanes by testing _phase_is_spillback_blocked directly.
    agent = FlowAgent({"spillback_threshold": 0.80})

    # Build a state where ALL outgoing lanes have occupancy 1.0
    out_full = [
        LaneState(lane_id=ol.lane_id, queue=20, vehicles=20, waiting_time=60.0, occupancy=1.0)
        for ol in s.out_lanes
    ]
    # Replace out_lanes with fully blocked lanes
    import dataclasses
    jammed_state = dataclasses.replace(s, out_lanes=tuple(out_full))

    all_allowed = [True] * jammed_state.num_phases
    # Since ALL phases are blocked by spillback, agent should hold current phase
    action = agent.act(jammed_state, [], all_allowed)
    assert action == jammed_state.phase
    assert "spillback-blocked" in agent.explain()


# ------------------------------------------------------------------ #
#  Observation encoding                                                #
# ------------------------------------------------------------------ #

def test_encode_state_shape_and_range():
    s = make_example_state()
    agent = FlowAgent({"max_queue": 25.0, "max_wait": 120.0})
    obs = agent.encode_state(s)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.ndim == 1

    # Features:
    # incoming lanes * 3  (queue, wait, occ)
    # + outgoing lanes * 1 (occupancy only)
    # + num_phases one-hot
    # + 1 time_in_phase
    # + num_phases downstream spillback signals
    expected_dim = (
        len(s.lanes) * 3
        + len(s.out_lanes) * 1
        + s.num_phases           # one-hot
        + 1                      # time_in_phase
        + s.num_phases           # per-phase downstream spillback
    )
    assert obs.shape[0] == expected_dim, (
        f"Expected {expected_dim} features, got {obs.shape[0]}"
    )

    # All normalized values must be between 0.0 and 1.0
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)
    assert not np.any(np.isnan(obs))

    # Check phase one-hot encoding
    one_hot_start = len(s.lanes) * 3 + len(s.out_lanes) * 1
    phase_slice = obs[one_hot_start : one_hot_start + s.num_phases]
    assert np.sum(phase_slice) == 1.0
    assert phase_slice[s.phase] == 1.0


def test_encode_state_spillback_signals_all_zero_when_clear():
    """Downstream spillback signals should be low (< 0.25) when outgoing lanes are clear."""
    s = make_example_state()
    # make_example_state out_lane occupancies are 0.10–0.20 (realistic, not jammed)
    agent = FlowAgent({})
    obs = agent.encode_state(s)
    # Per-phase spillback signals are at the end of the vector
    spillback_slice = obs[-(s.num_phases):]
    assert np.all(spillback_slice <= 0.25), (
        f"Expected low (non-jammed) spillback signals, got: {spillback_slice}"
    )


def test_encode_state_spillback_signals_high_when_blocked():
    """Downstream spillback signals should be high when outgoing lanes are jammed."""
    import dataclasses
    s = make_example_state()
    out_full = [
        LaneState(lane_id=ol.lane_id, queue=20, vehicles=20, waiting_time=60.0, occupancy=0.95)
        for ol in s.out_lanes
    ]
    jammed_state = dataclasses.replace(s, out_lanes=tuple(out_full))
    agent = FlowAgent({})
    obs = agent.encode_state(jammed_state)
    spillback_slice = obs[-(jammed_state.num_phases):]
    # At least one phase should show a high spillback signal
    assert np.any(spillback_slice >= 0.5)


# ------------------------------------------------------------------ #
#  RL model integration                                                #
# ------------------------------------------------------------------ #

def test_flow_act_with_mock_model_allowed():
    s = make_example_state()
    mock_model = MockModel(action=1)
    agent = FlowAgent({}, model=mock_model)
    all_allowed = [True] * s.num_phases

    action = agent.act(s, [], all_allowed)
    assert action == 1
    assert "RL model chose phase 1" in agent.explain()


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


def test_flow_act_with_mock_model_blocked_by_spillback():
    """RL model picks a phase whose downstream is fully jammed; should fall back to heuristic."""
    import dataclasses
    s = make_example_state()
    # Jam all downstream lanes
    out_full = [
        LaneState(lane_id=ol.lane_id, queue=20, vehicles=20, waiting_time=60.0, occupancy=1.0)
        for ol in s.out_lanes
    ]
    jammed_state = dataclasses.replace(s, out_lanes=tuple(out_full))

    mock_model = MockModel(action=0)
    agent = FlowAgent({"spillback_threshold": 0.80}, model=mock_model)
    all_allowed = [True] * jammed_state.num_phases

    # All downstream jammed → heuristic should hold current phase
    action = agent.act(jammed_state, [], all_allowed)
    # Should not have released into the jam; will hold current phase
    assert action == jammed_state.phase


def test_flow_act_with_failing_model_falls_back():
    s = make_example_state()
    failing_model = MockModel(action=0, fail=True)
    agent = FlowAgent({}, model=failing_model)
    all_allowed = [True] * s.num_phases

    # Should not raise exception, but cleanly fall back to heuristic
    action = agent.act(s, [], all_allowed)
    assert action == 0
    assert "heuristic" in agent.explain()
