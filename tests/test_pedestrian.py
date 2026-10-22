"""Unit tests for the pedestrian specialist and safety shield."""
from dataclasses import replace

from flowmind.agents.pedestrian import PedestrianAgent
from flowmind.contracts import Crossing, make_example_state
from flowmind.control.shield import SafetyShield


def test_pedestrian_claims_longest_waiting_crossing():
    state = make_example_state()
    agent = PedestrianAgent({"min_waiting": 1, "max_wait": 60})

    claim = agent.wants_control(state, [])

    assert claim is not None
    assert 0.0 < claim.urgency <= 1.0
    assert ":J1_c0" in claim.reason
    assert "80 s" in claim.reason


def test_pedestrian_stays_idle_below_threshold():
    state = replace(
        make_example_state(),
        crossings=(Crossing(":J1_c0", waiting=2, max_wait=59, on_crossing=0),),
    )
    agent = PedestrianAgent({"min_waiting": 1, "max_wait": 60})

    assert agent.wants_control(state, []) is None


def test_small_group_waits_before_hard_cap():
    state = replace(
        make_example_state(),
        crossings=(Crossing(":J1_c0", waiting=3, max_wait=59, on_crossing=0),),
    )
    agent = PedestrianAgent(
        {"min_waiting": 4, "max_wait": 5, "hard_wait": 60}
    )

    assert agent.wants_control(state, []) is None


def test_group_of_four_requests_control():
    state = replace(
        make_example_state(),
        time_in_phase=35,
        crossings=(Crossing(":J1_c0", waiting=4, max_wait=5, on_crossing=0),),
    )
    agent = PedestrianAgent(
        {"min_waiting": 4, "max_wait": 5, "hard_wait": 60}
    )

    assert agent.wants_control(state, []) is not None


def test_compatible_crossings_are_grouped_for_one_phase():
    state = replace(
        make_example_state(),
        time_in_phase=35,
        crossings=(
            Crossing(":J1_c0", waiting=2, max_wait=6, on_crossing=0),
            Crossing(":J1_c2", waiting=2, max_wait=5, on_crossing=0),
        ),
        phase_crossings=((":J1_c0", ":J1_c2"), (), (), ()),
    )
    agent = PedestrianAgent(
        {"min_waiting": 4, "max_wait": 5, "hard_wait": 60}
    )

    claim = agent.wants_control(state, [])

    assert claim is not None
    assert "4 pedestrians" in claim.reason


def test_group_waits_for_vehicle_service_window():
    state = replace(
        make_example_state(),
        time_in_phase=34,
        crossings=(Crossing(":J1_c0", waiting=4, max_wait=20, on_crossing=0),),
    )
    agent = PedestrianAgent(
        {
            "min_waiting": 4,
            "max_wait": 5,
            "hard_wait": 60,
            "batch_service_after": 35,
            "large_group_priority": 10,
        }
    )

    assert agent.wants_control(state, []) is None


def test_large_group_can_request_before_service_window():
    state = replace(
        make_example_state(),
        time_in_phase=10,
        crossings=(Crossing(":J1_c0", waiting=10, max_wait=5, on_crossing=0),),
    )
    agent = PedestrianAgent(
        {
            "min_waiting": 4,
            "max_wait": 5,
            "hard_wait": 60,
            "batch_service_after": 35,
            "large_group_priority": 10,
        }
    )

    assert agent.wants_control(state, []) is not None


def test_small_group_becomes_urgent_at_hard_cap():
    state = replace(
        make_example_state(),
        crossings=(Crossing(":J1_c0", waiting=1, max_wait=60, on_crossing=0),),
    )
    agent = PedestrianAgent(
        {"min_waiting": 4, "max_wait": 5, "hard_wait": 60}
    )

    claim = agent.wants_control(state, [])

    assert claim is not None
    assert claim.urgency > 0.75


def test_urgency_increases_with_wait_and_group_size():
    agent = PedestrianAgent(
        {"min_waiting": 4, "max_wait": 5, "hard_wait": 60}
    )
    lower_state = replace(
        make_example_state(),
        time_in_phase=35,
        crossings=(Crossing(":J1_c0", waiting=4, max_wait=5, on_crossing=0),),
    )
    higher_state = replace(
        make_example_state(),
        time_in_phase=35,
        crossings=(Crossing(":J1_c0", waiting=7, max_wait=45, on_crossing=0),),
    )

    lower = agent.wants_control(lower_state, [])
    higher = agent.wants_control(higher_state, [])

    assert lower is not None and higher is not None
    assert higher.urgency > lower.urgency


def test_pedestrian_selects_phase_for_crossing():
    state = make_example_state()
    agent = PedestrianAgent({"min_waiting": 1, "max_wait": 60})

    action = agent.act(state, [], [True] * state.num_phases)

    assert action == 0
    assert ":J1_c0" in agent.explain()


def test_pedestrian_falls_back_when_crossing_phase_is_blocked():
    state = make_example_state()
    agent = PedestrianAgent({"min_waiting": 1, "max_wait": 60})

    action = agent.act(state, [], [False, True, True, True])

    assert action == state.phase
    assert "not currently allowed" in agent.explain()


def make_shield():
    return SafetyShield(
        {"min_green": 10, "max_green": 60},
        {"ped_max_wait": 90, "ped_clearance_speed": 0.8},
    )


def test_shield_blocks_early_switch():
    state = replace(make_example_state(), time_in_phase=5)
    result = make_shield().check(state, 0)

    assert result.overridden
    assert result.action == state.phase


def test_shield_keeps_phase_serving_active_crossing():
    state = make_example_state()  # :J1_c1 is active and served by current phase 2
    allowed = make_shield().allowed_phases(state)

    assert allowed == [False, False, True, False]


def test_shield_forces_service_at_pedestrian_wait_cap():
    state = replace(
        make_example_state(),
        crossings=(Crossing(":J1_c0", waiting=3, max_wait=95, on_crossing=0),),
    )
    allowed = make_shield().allowed_phases(state)

    assert allowed == [True, False, False, False]


def test_shield_disallows_current_phase_at_max_green():
    state = replace(
        make_example_state(),
        time_in_phase=60,
        crossings=(),
    )
    allowed = make_shield().allowed_phases(state)

    assert not allowed[state.phase]
    assert any(allowed)


def test_vehicle_service_wins_when_pedestrian_priority_reaches_max_green():
    state = replace(
        make_example_state(),
        phase=0,
        time_in_phase=60,
        crossings=(Crossing(":J1_c0", waiting=8, max_wait=120, on_crossing=0),),
    )

    allowed = make_shield().allowed_phases(state)

    assert not allowed[state.phase]
    assert allowed == [False, True, True, True]
