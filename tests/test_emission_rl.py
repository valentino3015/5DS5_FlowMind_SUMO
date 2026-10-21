"""PPO emission reward, encoding, topology checks, and action safety."""
from dataclasses import replace
import numpy as np
import pytest
from flowmind.emission_rl import SafetyGate, encode, topology
from flowmind.contracts import JunctionState, LaneState, LaneEmissions
from scripts.train_emissions import environmental_reward


def states():
    return {"J": JunctionState("J",20,2,0,20,False,
        (LaneState("a",1,2,3,.1),LaneState("b",2,3,4,.1)),
        (LaneState("oa",0,0,0,0),LaneState("ob",0,0,0,0)),(),(),
        (("a",),("b",)),((),()),(("oa",),("ob",)),
        (LaneEmissions("a",10,2,5,1),LaneEmissions("b",20,3,10,2)))}


def test_reward_only_uses_four_quantities_with_fixed_scales():
    assert environmental_reward([10,20,30,40],[10,20,30,40],[1]*4)==-1
    assert environmental_reward([0]*4,[10,20,30,40],[1]*4)==0
    assert environmental_reward([20,40,60,80],[10,20,30,40],[1]*4)==-2


def test_observations_ignore_queue_wait_and_vehicle_count():
    st=states()
    manifest={"topology":topology(st),"scales_mg_s":[10,2,5,1]}
    initial=encode(st,manifest)
    changed={"J":replace(st["J"],lanes=tuple(replace(l,queue=100,waiting_time=1000,vehicles=100) for l in st["J"].lanes))}
    assert np.array_equal(initial,encode(changed,manifest))
    assert initial.shape==(11,)
    with pytest.raises(ValueError,match="réseau"):
        encode({"different":st["J"]},manifest)


def test_model_actions_respect_min_green_saturation_and_max_green():
    st=states()
    gate=SafetyGate({})
    assert gate.apply({"J":replace(st["J"],time_in_phase=1)}, {"J":1})[0]["J"]==0
    assert gate.apply({"J":replace(st["J"],time_in_phase=60)}, {"J":0})[0]["J"]==1
    blocked=replace(st["J"],out_lanes=(st["J"].out_lanes[0],replace(st["J"].out_lanes[1],occupancy=.95)))
    assert gate.apply({"J":blocked},{"J":1})[0]["J"]==0


def test_four_weight_validation_still_allows_model_method():
    from flowmind.runtime import RuntimeControl
    control=RuntimeControl("emission_model")
    assert control.snapshot()["method"]=="emission_model"


def test_trained_model_rejects_training_seed_and_other_scenario():
    from flowmind.emission_rl import MODEL_DIR, EmissionPolicy
    from flowmind.evaluation import run_experiment
    if not (MODEL_DIR / "best.zip").exists():
        pytest.skip("No smoke model installed")
    with pytest.raises(ValueError,match="Seed réservée"):
        run_experiment("emission_model",seed=0,seconds=10)
    with pytest.raises(ValueError,match="autre scénario"):
        EmissionPolicy("cologne3",{})


def test_evaluation_does_not_fall_back_when_explicit_model_is_missing(tmp_path):
    from flowmind.evaluation import run_experiment
    with pytest.raises(ValueError, match="Modèle émissions absent"):
        run_experiment("emission_model", seconds=10, seed=10010,
                       model_directory=tmp_path)


@pytest.mark.parametrize("score_values,teleports", [([-100., -120., -80.], [0,0,0]), ([-100., -70., -80.], [0,2,0])])
def test_validation_preserves_parent_until_a_better_checkpoint(monkeypatch, tmp_path, score_values, teleports):
    from types import SimpleNamespace
    import scripts.train_emissions as training
    scores, teleport_counts = iter(score_values), iter(teleports)

    class ValidationEnv:
        def __init__(self, *args):
            self.total_reward = next(scores)
            self.episode_stats = {"teleports": next(teleport_counts), "collisions": 0}

        def reset(self):
            return np.zeros(1), {}

        def step(self, action):
            return np.zeros(1), 0, False, True, {}

        def close(self):
            pass

    monkeypatch.setattr(training, "EmissionTrainingEnv", ValidationEnv)
    raw = SimpleNamespace(close=lambda: None, episode_rows=[])
    callback = training.Validate(raw, {}, {}, [1000], tmp_path, 64)
    callback.model = SimpleNamespace(
        predict=lambda obs, deterministic: (0, None),
        save=lambda path: path.write_text(str(callback.num_timesteps)))
    for step in (128, 1152, 2176):
        callback.num_timesteps = step
        callback.assess()
        if step == 1152:
            assert (tmp_path / "best.zip").read_text() == "128"
    assert callback.best_timesteps == 2176
    assert (tmp_path / "best.zip").read_text() == "2176"


def test_rl_limits_intervene_before_teleport_without_changing_reference():
    import yaml
    from flowmind.env import ROOT
    from flowmind.emission_rl import training_safety, inference_safety
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    limits = training_safety(config)
    assert config["fixed_time"]["green"] == 30
    assert limits["max_red"] == 30
    assert limits["max_green"] == 35
    resolved = inference_safety({"safety": limits}, {"max_red": 90, "max_green": 60, "min_green": 10})
    assert resolved == limits
    with pytest.raises(ValueError, match="incompatible"):
        inference_safety({"safety": limits}, {"min_green": 60})
    config["emission_rl"]["safety"]["max_red"] = 55
    with pytest.raises(ValueError, match="60 s"):
        training_safety(config)


def test_rl_serves_queued_red_direction_before_teleport_deadline():
    gate = SafetyGate({"max_red": 30, "max_green": 35})
    initial = states()
    assert gate.apply(initial, {"J":0})[0]["J"] == 0
    later = {"J": replace(initial["J"], time=50, time_in_phase=30)}
    assert gate.apply(later, {"J":0})[0]["J"] == 1


def test_doubled_pm_weight_changes_only_environmental_reward():
    assert environmental_reward([0,0,0,10], [10]*4, [1,1,1,2]) == -.4
    assert environmental_reward([0,0,0,10], [10]*4, [1]*4) == -.25
