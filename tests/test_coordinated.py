"""Coordinated duration action: one green duration for all junctions; departure noise."""
import sys
from dataclasses import replace
import pytest
from flowmind.emission_rl import topology
from test_emission_rl import states


def grid():
    st = states()
    return {"J": st["J"], "K": replace(st["J"], tls_id="K", time_in_phase=12)}


def manifest(coordinated=True):
    st = grid()
    return {"topology": topology(st), "action_mode": "duration", "green_durations_s": [15, 20, 25, 30],
            "coordinated_action": coordinated}


def test_one_duration_is_applied_to_every_junction():
    from flowmind.emission_rl import decode_actions
    st = grid()  # J has 20 s of green, K 12 s
    assert decode_actions(st, [0], manifest()) == {"J": 1, "K": 0}  # 15 s: only J switches
    assert decode_actions(st, [1], manifest()) == {"J": 1, "K": 0}  # 20 s
    assert decode_actions(st, [2], manifest()) == {"J": 0, "K": 0}  # 25 s: nobody switches
    with pytest.raises(ValueError, match="incompatible"):
        decode_actions(st, [0], manifest(coordinated=False))
    assert decode_actions(st, [0, 2], manifest(coordinated=False)) == {"J": 1, "K": 0}


def test_coordinated_environment_has_a_single_action():
    from scripts.train_emissions import EmissionTrainingEnv
    assert EmissionTrainingEnv({}, manifest(), [1]).action_space.nvec.tolist() == [4]
    assert EmissionTrainingEnv({}, manifest(False), [1]).action_space.nvec.tolist() == [4, 4]


def test_one_missing_departure_is_noise_two_are_a_loss():
    from scripts.train_emissions import compare_validation, tolerant_selection
    baseline = {k: {"emissions_mg": [100.] * 4, "departed": 300, "arrived": 250} for k in "ab"}

    def run(departed_b):
        measured = {"a": {"emissions_mg": [93., 93., 93., 101.], "departed": 300, "arrived": 250},
                    "b": {"emissions_mg": [93., 93., 93., 101.], "departed": departed_b, "arrived": 250}}
        return tolerant_selection(compare_validation(measured, baseline), .02)[0]
    assert compare_validation({"a": {**baseline["a"], "departed": 299}}, baseline)["departure_gap"] == 1
    assert run(300) and run(299) and not run(298)


def test_coordinated_requires_new_duration_policy(monkeypatch, tmp_path, capsys):
    import scripts.train_emissions as training
    monkeypatch.setattr(sys, "argv", ["train_emissions.py", "--output", str(tmp_path / "m"), "--coordinated"])
    with pytest.raises(SystemExit) as exit_info:
        training.main()
    assert exit_info.value.code == 2
    assert "--coordinated" in capsys.readouterr().err
