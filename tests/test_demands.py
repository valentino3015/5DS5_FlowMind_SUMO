"""Varied demands: separated pools, reproducible draws, paired validation."""
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from flowmind.demands import (case_key, check_disjoint, file_sha256, load_demands,
                              validation_cases)


def make_pool(root, files):
    """files: {relative path: (role, content)}; writes routes and index.json."""
    entries = []
    for rel, (role, content) in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        entries.append({"role": role, "file": rel, "sha256": file_sha256(path)})
    (root / "index.json").write_text(json.dumps({"files": entries}), encoding="utf-8")
    return root


def test_load_demands_accepts_only_listed_unmodified_files_of_the_role(tmp_path):
    root = make_pool(tmp_path, {"train/a.rou.xml": ("train", "<routes>a</routes>"),
                                "train/b.rou.xml": ("train", "<routes>b</routes>"),
                                "test/t.rou.xml": ("test", "<routes>t</routes>")})
    assert [p.name for p in load_demands(root / "train", "train")] == ["a.rou.xml", "b.rou.xml"]
    with pytest.raises(ValueError, match="réservée"):
        load_demands(root / "test", "train")
    (root / "train" / "a.rou.xml").write_text("<routes>changed</routes>", encoding="utf-8")
    with pytest.raises(ValueError, match="modifiée"):
        load_demands(root / "train", "train")
    (root / "train" / "a.rou.xml").write_text("<routes>a</routes>", encoding="utf-8")
    (root / "train" / "c.rou.xml").write_text("<routes>c</routes>", encoding="utf-8")
    with pytest.raises(ValueError, match="absente"):
        load_demands(root / "train", "train")


def test_test_file_copied_into_train_folder_is_refused(tmp_path):
    root = make_pool(tmp_path, {"train/a.rou.xml": ("train", "<routes>a</routes>"),
                                "test/t.rou.xml": ("test", "<routes>t</routes>")})
    (root / "train" / "t.rou.xml").write_text("<routes>t</routes>", encoding="utf-8")
    with pytest.raises(ValueError):
        load_demands(root / "train", "train")


def test_identical_route_in_two_pools_is_refused(tmp_path):
    a, b = tmp_path / "a.rou.xml", tmp_path / "b.rou.xml"
    a.write_text("same", encoding="utf-8")
    b.write_text("same", encoding="utf-8")
    with pytest.raises(ValueError, match="partagée"):
        check_disjoint([a], [b])
    check_disjoint([a], [])


def test_validation_cases_use_each_demand_once_with_rotating_seeds(tmp_path):
    demands = [tmp_path / f"v{i}.rou.xml" for i in range(3)]
    cases = validation_cases([1000, 1001], demands)
    assert [(s, r.name) for s, r in cases] == [(1000, "v0.rou.xml"), (1001, "v1.rou.xml"), (1000, "v2.rou.xml")]
    assert validation_cases([5, 6]) == [(5, None), (6, None)]
    assert case_key(1000, demands[0]) == "v0@1000" and case_key(1000) == 1000


class FakeSumo:
    created = []

    def __init__(self, config=None, route_file=None, seed=None, use_gui=False):
        FakeSumo.created.append((route_file, seed))
        self.metrics_callback = None

    def reset(self):
        return {}

    def close(self):
        pass


def episodes(monkeypatch, *args):
    import scripts.train_emissions as training
    FakeSumo.created = []
    monkeypatch.setattr(training, "SumoCorridorEnv", FakeSumo)
    monkeypatch.setattr(training, "encode", lambda states, manifest: np.zeros(1))
    monkeypatch.setattr(training, "SafetyGate", lambda params: None)
    manifest = {"topology": [{"phases": 2}], "safety": {}}
    env = training.EmissionTrainingEnv({}, manifest, *args)
    for _ in range(6):
        env.reset()
    return [(r.name if r else None, s) for r, s in FakeSumo.created]


def test_training_draws_a_reproducible_demand_per_episode(monkeypatch, tmp_path):
    demands = [tmp_path / f"d{i}.rou.xml" for i in range(5)]
    first = episodes(monkeypatch, [9, 10], demands, True, 7)
    assert first == episodes(monkeypatch, [9, 10], demands, True, 7)
    assert first != episodes(monkeypatch, [9, 10], demands, True, 8)
    assert len({name for name, _ in first}) > 1
    assert [s for _, s in first] == [9, 10] * 3


def test_validation_mode_keeps_demand_order_and_old_behaviour_without_demands(monkeypatch, tmp_path):
    demands = [tmp_path / f"v{i}.rou.xml" for i in range(3)]
    assert episodes(monkeypatch, [1, 2, 1], demands, False)[:3] == [("v0.rou.xml", 1), ("v1.rou.xml", 2), ("v2.rou.xml", 1)]
    assert episodes(monkeypatch, [4, 5])[:2] == [(None, 4), (None, 5)]


def test_validation_compares_each_demand_with_its_own_reference(monkeypatch, tmp_path):
    import scripts.train_emissions as training
    demands = [tmp_path / "v0.rou.xml", tmp_path / "v1.rou.xml"]
    calls = []

    class ValidationEnv:
        def __init__(self, config, manifest, seeds, routes, sample):
            assert sample is False and [r.name for r in routes] == ["v0.rou.xml", "v1.rou.xml"]
            self.total_reward, self.episode_stats = -1., {"teleports": 0, "collisions": 0}
            self.env = SimpleNamespace(close=lambda: None)

        def reset(self):
            return np.zeros(1), {}

        def step(self, action):
            return np.zeros(1), 0, False, True, {}

        def close(self):
            pass

    def official(path):
        calls.append(path.parent.name)
        return {"emissions_mg": [90., 90., 90., 90.], "departed": 10, "arrived": 10}

    monkeypatch.setattr(training, "EmissionTrainingEnv", ValidationEnv)
    monkeypatch.setattr(training, "official_emissions", official)
    baseline = {"v0@1000": {"emissions_mg": [100.] * 4, "departed": 10, "arrived": 10},
                "v1@1001": {"emissions_mg": [100.] * 4, "departed": 10, "arrived": 10}}
    raw = SimpleNamespace(close=lambda: None, episode_rows=[])
    callback = training.Validate(raw, {}, {}, [1000, 1001], tmp_path, 64, baseline=baseline, demands=demands)
    callback.model = SimpleNamespace(predict=lambda obs, deterministic: (0, None), save=lambda path: None)
    callback.num_timesteps = 64
    callback.assess()
    assert calls == ["seed_v0@1000", "seed_v1@1001"]
    assert callback.best_comparison["all_emissions_lower"] and callback.best_comparison["co2_ratio"] == pytest.approx(.9)


def run_main(monkeypatch, *argv):
    import scripts.train_emissions as training
    monkeypatch.setattr(sys, "argv", ["train_emissions.py", *map(str, argv)])
    with pytest.raises(SystemExit) as exit_info:
        training.main()
    return exit_info.value.code


def test_training_refuses_test_demands_and_unpaired_demand_options(monkeypatch, tmp_path, capsys):
    root = make_pool(tmp_path / "demand", {"train/a.rou.xml": ("train", "a"),
                                           "validation/v.rou.xml": ("validation", "v"),
                                           "test/t.rou.xml": ("test", "t")})
    out = tmp_path / "model"
    assert run_main(monkeypatch, "--output", out, "--train-demands", root / "test",
                    "--validation-demands", root / "validation") == 2
    assert "réservée" in capsys.readouterr().err
    assert run_main(monkeypatch, "--output", out, "--train-demands", root / "train") == 2
    assert not out.exists()


def test_ppo_hyperparameters_are_refused_when_resuming(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, "--output", tmp_path / "m", "--resume", tmp_path / "parent", "--n-steps", 1024) == 2
    assert "reprise" in capsys.readouterr().err


def test_generated_pools_cover_every_load_with_disjoint_seeds():
    from scripts.generate_demands import LOADS, MAX_ATTEMPTS, SEED_BASE, recipe
    assert {recipe("validation", i)["load"] for i in range(len(LOADS))} == {name for name, _ in LOADS}
    seeds = {role: {SEED_BASE[role] + 10 * i + a for i in range(count) for a in range(MAX_ATTEMPTS)}
             for role, count in (("train", 40), ("validation", 6), ("test", 10))}
    assert not (seeds["train"] & seeds["validation"] or seeds["train"] & seeds["test"]
                or seeds["validation"] & seeds["test"])


def test_reward_uses_the_fixed_time_rates_of_the_episode_demand(monkeypatch):
    import scripts.train_emissions as training
    manifest = {"scales_mg_s": [10.] * 4, "demand_scales_mg_s": {"heavy.rou.xml": [40.] * 4}}
    env = SimpleNamespace(manifest=manifest, episode_demand="heavy.rou.xml")
    assert training.EmissionTrainingEnv.reward_scales(env) == [40.] * 4
    env.episode_demand = "validation_00.rou.xml"
    assert training.EmissionTrainingEnv.reward_scales(env) == [10.] * 4
    # Same traffic as the reference: ratio 1 whatever the demand's load.
    heavy = training.worst_interval_cost([40.] * 4, [40.] * 4)
    light = training.worst_interval_cost([10.] * 4, [10.] * 4)
    assert heavy == light == pytest.approx(1.)


def test_per_demand_scales_have_one_entry_per_training_file(monkeypatch, tmp_path):
    import scripts.train_emissions as training
    runs = []

    def fake_run(config, seed, route=None):
        runs.append((seed, route.name))
        rate = 100. if "heavy" in route.name else 0.
        return [{k: rate for k in training.SAMPLE_KEYS}] * 3, None

    monkeypatch.setattr(training, "fixed_time_samples", fake_run)
    scales = training.demand_scales({}, 9, [tmp_path / "heavy.rou.xml", tmp_path / "light.rou.xml"])
    assert scales == {"heavy.rou.xml": [100.] * 4, "light.rou.xml": [1.] * 4}  # floor avoids division by zero
    assert runs == [(9, "heavy.rou.xml"), (9, "light.rou.xml")]


@pytest.mark.parametrize("neutral,expected", [(False, [.41, .41]), (True, [.25, .25])])
def test_duration_policy_initialization(neutral, expected):
    import torch
    from scripts.train_emissions import init_duration_policy
    policy = SimpleNamespace(action_net=torch.nn.Linear(3, 8))
    init_duration_policy(policy, neutral)
    probs = torch.softmax(policy.action_net(torch.randn(1, 3)).reshape(-1, 4), 1)
    assert probs[:, 2].tolist() == pytest.approx(expected, abs=.01)
    assert probs.argmax(1).tolist() == [2, 2] or neutral


def test_per_demand_reward_and_neutral_init_option_checks(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, "--output", tmp_path / "m", "--per-demand-reward") == 2
    assert "--train-demands" in capsys.readouterr().err
    assert run_main(monkeypatch, "--output", tmp_path / "m", "--neutral-init") == 2
    assert "duration" in capsys.readouterr().err
