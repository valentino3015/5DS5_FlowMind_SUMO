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


def test_validation_compares_all_four_emissions_and_traffic_to_reference():
    from scripts.train_emissions import compare_validation
    reference = {1000: {"emissions_mg":[100.,10.,50.,2.], "arrived":100, "departed":120}}
    all_better = {1000: {"emissions_mg":[90.,9.,45.,1.8], "arrived":100, "departed":120}}
    compared = compare_validation(all_better, reference)
    assert compared["all_emissions_lower"]
    assert compared["worst_ratio"] == 1
    worse_pm = {1000: all_better[1000] | {"emissions_mg":[80.,8.,40.,2.4]}}
    assert not compare_validation(worse_pm, reference)["all_emissions_lower"]
    assert compare_validation(worse_pm, reference)["worst_ratio"] == pytest.approx(1.2)
    blocked = {1000: all_better[1000] | {"arrived":80}}
    assert compare_validation(blocked, reference)["worst_ratio"] == 1.25


def test_official_validation_totals_include_unfinished_trips(tmp_path):
    from scripts.train_emissions import official_emissions
    path = tmp_path / "tripinfo.xml"
    path.write_text('<tripinfos><tripinfo arrival="10"><emissions CO2_abs="10" NOx_abs="2" fuel_abs="3" PMx_abs="1"/></tripinfo>'
                    '<tripinfo arrival="-1"><emissions CO2_abs="20" NOx_abs="4" fuel_abs="6" PMx_abs="2"/></tripinfo></tripinfos>')
    data = official_emissions(path)
    assert data == {"emissions_mg":[30.,6.,9.,3.], "departed":2, "arrived":1}


def test_sampling_inference_requires_seed_and_repeats_learned_probabilities(tmp_path, monkeypatch):
    import json
    from stable_baselines3 import PPO
    from flowmind.emission_rl import EmissionPolicy
    params = {"min_green":10, "max_green":60, "max_red":90, "blocked_occupancy":.8}
    manifest = {"scenario":"3x3grid", "topology":topology(states()), "scales_mg_s":[10,2,5,1],
                "safety":params, "deterministic_inference":False, "policy_seed_offset":100000}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    calls=[]
    class FakeModel:
        def set_random_seed(self, seed):
            self.rng=np.random.default_rng(seed)
        def predict(self, obs, deterministic):
            calls.append(deterministic)
            return np.array([0 if deterministic else self.rng.choice([0,1],p=[.3,.7])]),None
    monkeypatch.setattr(PPO,'load',lambda *args,**kwargs:FakeModel())
    with pytest.raises(ValueError,match="seed"):
        EmissionPolicy('3x3grid',params,directory=tmp_path)
    first=EmissionPolicy('3x3grid',params,directory=tmp_path,seed=1002)
    second=EmissionPolicy('3x3grid',params,directory=tmp_path,seed=1002)
    assert first.policy_seed==101002
    a=[first.act_all(states())[0]['J'] for _ in range(25)]
    b=[second.act_all(states())[0]['J'] for _ in range(25)]
    assert a==b and len(set(a))==2
    assert not any(calls)
    manifest.pop('deterministic_inference')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    legacy=EmissionPolicy('3x3grid',params,directory=tmp_path)
    assert legacy.deterministic and legacy.act_all(states())[0]['J']==0
