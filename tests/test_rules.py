"""Rules optimizer: decision constraints, pressure, hysteresis and starvation."""
from dataclasses import replace
import pytest
from flowmind.agents.rules import RuleBasedAgent
from flowmind.contracts import JunctionState, LaneState, LaneEmissions


def state(queue_a=1, queue_b=10, time=20, green=20, out_b=0, emission_a=1, emission_b=10):
    return JunctionState("J", time, 2, 0, green, False,
        (LaneState("a", queue_a, queue_a, queue_a*5, .1),
         LaneState("b", queue_b, queue_b, queue_b*5, .1)),
        (LaneState("oa", 0, 0, 0, 0), LaneState("ob", 0, 0, 0, out_b)),
        (), (), (("a",), ("b",)), ((), ()), (("oa",), ("ob",)),
        (LaneEmissions("a", *([emission_a]*4)), LaneEmissions("b", *([emission_b]*4))))


def test_priority_goes_to_emission_burden():
    a = RuleBasedAgent()
    assert a.act(state(), [], [True, True]) == 1
    assert a.explain()


def test_minimum_green_and_transition_prevent_switch():
    a = RuleBasedAgent()
    assert a.act(state(green=3), [], [True, True]) == 0
    assert a.act(replace(state(), in_transition=True), [], [True, True]) == 0


def test_blocked_exit_and_disallowed_phase_are_not_selected():
    assert RuleBasedAgent().act(state(out_b=.95), [], [True, True]) == 0
    assert RuleBasedAgent().act(state(), [], [True, False]) == 0
    assert RuleBasedAgent().act(state(green=1), [], [False, True]) == 1


def test_small_gain_does_not_cause_repeated_switches():
    assert RuleBasedAgent().act(state(emission_a=5, emission_b=5.1), [], [True, True]) == 0


def test_starvation_overrides_higher_current_demand():
    a = RuleBasedAgent({"max_red": 30})
    assert a.act(state(queue_a=30, queue_b=1, emission_a=30, emission_b=1, time=20), [], [True, True]) == 0
    assert a.act(state(queue_a=30, queue_b=1, emission_a=30, emission_b=1, time=51), [], [True, True]) == 1


def test_maximum_green_forces_available_alternative():
    assert RuleBasedAgent().act(state(queue_a=30, queue_b=1, emission_a=30, emission_b=1, green=60), [], [True, True]) == 1


def test_no_authorized_phase_is_an_explicit_error():
    with pytest.raises(ValueError):
        RuleBasedAgent().act(state(), [], [False, False])


def test_reset_discards_old_red_ages():
    a = RuleBasedAgent({"max_red": 30})
    a.act(state(queue_a=30, queue_b=1, emission_a=30, emission_b=1, time=100), [], [True, True])
    assert a.act(state(queue_a=30, queue_b=1, emission_a=30, emission_b=1, time=20), [], [True, True]) == 0


def test_environment_uses_fractional_occupancy(monkeypatch):
    from flowmind.env import SumoCorridorEnv, traci
    env = SumoCorridorEnv()
    env.phase_out_lanes = {"J": [["out"]]}
    monkeypatch.setattr(traci.lane, "getLastStepOccupancy", lambda lane: 20)
    assert not env._is_downstream_blocked("J", 0)
    monkeypatch.setattr(traci.lane, "getLastStepOccupancy", lambda lane: 90)
    assert env._is_downstream_blocked("J", 0)


def test_score_does_not_use_queues_waiting_or_vehicle_counts():
    agent = RuleBasedAgent()
    first = state(emission_a=30, emission_b=1)
    changed = replace(first, lanes=tuple(replace(l, queue=1000, vehicles=2000, waiting_time=9999) for l in first.lanes))
    assert agent.emission_scores(first) == agent.emission_scores(changed)
    assert agent.act(changed, [], [True, True]) == 0


@pytest.mark.parametrize('metric', ['co2', 'nox', 'fuel', 'pm'])
def test_each_environmental_weight_can_determine_the_choice(metric):
    rates = (LaneEmissions('a', 100, 1, 100, 1), LaneEmissions('b', 1, 100, 1, 100))
    agent = RuleBasedAgent({k+'_weight': int(k==metric) for k in ('co2','nox','fuel','pm')})
    st = replace(state(), lane_emissions=rates)
    assert agent.act(st, [], [True, True]) == (0 if metric in ('co2','fuel') else 1)


def test_zero_emissions_and_missing_data_are_handled():
    assert RuleBasedAgent().act(state(emission_a=0, emission_b=0), [], [True, True]) == 0
    with pytest.raises(ValueError, match='manquantes'):
        RuleBasedAgent().act(replace(state(), lane_emissions=()), [], [True, True])


def test_normalization_is_invariant_to_scale_of_each_pollutant():
    a = state()
    b = replace(a, lane_emissions=tuple(replace(e, co2=e.co2*10000, nox=e.nox*.001) for e in a.lane_emissions))
    assert RuleBasedAgent().emission_scores(a) == pytest.approx(RuleBasedAgent().emission_scores(b))
