"""Integration tests for the global SUMO environment (flowmind/env.py).

Run with: pytest tests/test_env.py
"""
import pytest
from pathlib import Path

from flowmind.agents.baselines import FixedTime
from flowmind.contracts import JunctionState
from flowmind.env import SumoCorridorEnv


@pytest.fixture(scope="module")
def sumo_env():
    env = SumoCorridorEnv()
    yield env
    env.close()


def test_env_start_and_tls_discovery(sumo_env):
    states = sumo_env.reset()
    assert isinstance(states, dict)
    assert len(sumo_env.tls_ids) >= 1

    for tls_id in sumo_env.tls_ids:
        state = states[tls_id]
        assert isinstance(state, JunctionState)
        assert state.tls_id == tls_id
        assert 0 <= state.phase < state.num_phases
        assert state.time_in_phase == 0.0
        assert not state.in_transition
        assert isinstance(state.lanes, tuple)
        assert isinstance(state.out_lanes, tuple)
        assert len(state.phase_lanes) == state.num_phases


def test_env_step_keep_phase(sumo_env):
    states = sumo_env.reset()
    actions = {tls: states[tls].phase for tls in sumo_env.tls_ids}

    next_states, done, info = sumo_env.step(actions)
    assert not done
    assert isinstance(info, dict)
    assert "time" in info

    for tls in sumo_env.tls_ids:
        st = next_states[tls]
        assert st.phase == actions[tls]
        assert st.time_in_phase == float(sumo_env.delta_time)
        assert not st.in_transition


def test_env_step_phase_transition(sumo_env):
    states = sumo_env.reset()
    tls = sumo_env.tls_ids[0]
    curr_phase = states[tls].phase
    num_phases = states[tls].num_phases

    if num_phases > 1:
        target_phase = (curr_phase + 1) % num_phases
        actions = {t: states[t].phase for t in sumo_env.tls_ids}
        actions[tls] = target_phase

        next_states, done, info = sumo_env.step(actions)
        assert not done
        assert next_states[tls].phase == target_phase
        assert next_states[tls].time_in_phase == 0.0


def test_env_movements_for_max_pressure(sumo_env):
    tls = sumo_env.tls_ids[0]
    movements = sumo_env.get_phase_movements(tls)
    assert isinstance(movements, dict)
    for p, moves in movements.items():
        assert isinstance(moves, list)
        for move in moves:
            assert len(move) == 2
            assert isinstance(move[0], str) and isinstance(move[1], str)


def test_env_runs_with_fixed_time_agent(sumo_env):
    states = sumo_env.reset()
    agent = FixedTime({"green": 10})
    allowed = [True] * 10

    for _ in range(3):
        actions = {}
        for tls in sumo_env.tls_ids:
            st = states[tls]
            act = agent.act(st, [], allowed[:st.num_phases])
            actions[tls] = act

        states, done, info = sumo_env.step(actions)
        assert not done
