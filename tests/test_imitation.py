"""Behaviour cloning of the rules agent: supervised fit and option checks."""
import sys
import gymnasium as gym
import numpy as np
import pytest


class Spaces(gym.Env):
    observation_space = gym.spaces.Box(0, np.inf, (6,), np.float32)
    action_space = gym.spaces.MultiDiscrete([3, 3])

    def reset(self, seed=None, options=None):
        return np.zeros(6, np.float32), {}

    def step(self, action):
        return np.zeros(6, np.float32), 0., True, False, {}


def test_fit_reproduces_a_teacher_that_picks_the_highest_emission_phase():
    from stable_baselines3 import PPO
    from scripts.imitate_rules import accuracy, fit
    rng = np.random.default_rng(0)
    observations = rng.random((1500, 6)).astype(np.float32)
    # Like the rules agent: per junction, the phase with the largest emission share.
    labels = np.stack([observations[:, :3].argmax(1), observations[:, 3:].argmax(1)], 1)
    model = PPO("MlpPolicy", Spaces(), seed=0, device="cpu", policy_kwargs={"net_arch": [64, 64]})
    assert accuracy(model, observations, labels) < .6
    assert fit(model, observations, labels, epochs=40, learning_rate=3e-3) > .95


def test_options_keep_training_and_validation_seeds_apart(monkeypatch, capsys):
    import scripts.imitate_rules as imitation
    monkeypatch.setattr(sys, "argv", ["imitate_rules.py", "--train-seeds", "1", "--validation-seeds", "1"])
    with pytest.raises(SystemExit) as exit_info:
        imitation.main()
    assert exit_info.value.code == 2
    assert "distinctes" in capsys.readouterr().err


def test_output_must_be_new(tmp_path, monkeypatch):
    import scripts.imitate_rules as imitation
    (tmp_path / "keep.txt").write_text("x")
    monkeypatch.setattr(sys, "argv", ["imitate_rules.py", "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="nouveau ou vide"):
        imitation.main()
