"""PM-tolerant objective: CO2/NOx/fuel first, PM within a tolerance, 20 s target."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest


def test_pm_within_tolerance_costs_nothing_and_excess_is_penalized():
    from scripts.train_emissions import pm_tolerant_cost
    scales = [10., 10., 10., 10.]
    assert pm_tolerant_cost([9., 9., 9., 10.2], scales, .02, 4.) == pytest.approx(.9)
    assert pm_tolerant_cost([9., 9., 9., 11.], scales, .02, 4.) == pytest.approx(.9 + 4 * .08)
    # Lower CO2/NOx/fuel with tolerated PM beats the reference.
    assert pm_tolerant_cost([9.3, 9.2, 9.3, 10.17], scales, .02, 4.) < pm_tolerant_cost([10.] * 4, scales, .02, 4.)


def comparison(co2, pm, arrivals=1.):
    return {"co2_ratio": co2, "nox_ratio": co2, "fuel_ratio": co2, "pm_ratio": pm,
            "arrival_shortfall_ratio": arrivals, "departure_shortfall_ratio": 1., "departure_gap": 0}


def test_selection_prefers_feasible_pm_then_lowest_co2_nox_fuel():
    from scripts.train_emissions import tolerant_selection
    feasible_small = tolerant_selection(comparison(.98, 1.01), .02)
    feasible_large = tolerant_selection(comparison(.93, 1.017), .02)
    pm_too_high = tolerant_selection(comparison(.90, 1.03), .02)
    fewer_arrivals = tolerant_selection(comparison(.90, 1.0, arrivals=1.01), .02)
    assert feasible_large > feasible_small > pm_too_high
    assert feasible_small > fewer_arrivals


def test_validation_reports_the_20_s_target_and_selects_with_tolerance(monkeypatch, tmp_path):
    import scripts.train_emissions as training
    results = iter([[95., 95., 95., 101.], [91., 91., 91., 104.], [92., 92., 92., 101.5]])

    class ValidationEnv:
        def __init__(self, *args):
            self.total_reward, self.episode_stats = -1., {"teleports": 0, "collisions": 0}
            self.env = SimpleNamespace(close=lambda: None)

        def reset(self):
            return np.zeros(1), {}

        def step(self, action):
            return np.zeros(1), 0, False, True, {}

        def close(self):
            pass

    monkeypatch.setattr(training, "EmissionTrainingEnv", ValidationEnv)
    monkeypatch.setattr(training, "official_emissions",
                        lambda path: {"emissions_mg": next(results), "departed": 10, "arrived": 10})
    classic = {1000: {"emissions_mg": [100.] * 4, "departed": 10, "arrived": 10}}
    green20 = {1000: {"emissions_mg": [93., 93., 93., 101.7], "departed": 10, "arrived": 10}}
    raw = SimpleNamespace(close=lambda: None, episode_rows=[])
    callback = training.Validate(raw, {}, {"reward_mode": "pm_tolerant", "pm_tolerance": .02}, [1000],
                                 tmp_path, 64, baseline=classic, target=green20)
    callback.model = SimpleNamespace(predict=lambda obs, deterministic: (0, None),
                                     save=lambda path: path.write_text(str(callback.num_timesteps)))
    for step in (0, 64, 128):
        callback.num_timesteps = step
        callback.assess()
    # Step 64 has the lowest CO2 but PM +4 %: rejected. Step 128 beats the 20 s target.
    assert callback.best_timesteps == 128
    assert callback.best_comparison["beats_target"]
    assert callback.best_comparison["target_co2_ratio"] == pytest.approx(92 / 93)
    assert not callback.rows[0]["beats_target"]


@pytest.mark.parametrize("durations,expected", [([15, 20, 25, 30], [.195, .195, .195, .414]), ([10, 15, 20], [1/3] * 3)])
def test_initialization_follows_configured_durations(durations, expected):
    import torch
    from scripts.train_emissions import init_duration_policy
    policy = SimpleNamespace(action_net=torch.nn.Linear(3, 2 * len(durations)))
    init_duration_policy(policy, False, durations)
    probs = torch.softmax(policy.action_net(torch.randn(1, 3)).reshape(-1, len(durations)), 1)
    assert probs[0].tolist() == pytest.approx(expected, abs=.01)


def run_main(monkeypatch, *argv):
    import scripts.train_emissions as training
    monkeypatch.setattr(sys, "argv", ["train_emissions.py", *map(str, argv)])
    with pytest.raises(SystemExit) as exit_info:
        training.main()
    return exit_info.value.code


def test_green_durations_require_new_duration_policy(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, "--output", tmp_path / "m", "--green-durations", "15", "20") == 2
    assert "duration" in capsys.readouterr().err
