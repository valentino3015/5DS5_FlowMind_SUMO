"""Runtime validation and real SUMO switching without resetting simulation time."""
import json
import csv
from pathlib import Path
import pytest
from flowmind.runtime import RuntimeControl


def test_control_validation_and_revision_are_atomic():
    c = RuntimeControl("rules")
    assert c.snapshot()["revision"] == 0
    c.update("rules", {"co2_weight": 3})
    assert c.snapshot()["revision"] == 1
    assert c.update("rules", {"co2_weight": 3})["revision"] == 1
    for values in ({"co2_weight": float("nan")}, {"unknown": 3}, {"min_green": 100, "max_green": 10}):
        with pytest.raises(ValueError):
            c.update("fixed_time", values)
        assert c.snapshot()["method"] == "rules"
        assert c.snapshot()["revision"] == 1
    with pytest.raises(ValueError, match="poids environnemental"):
        c.update("rules", {key + "_weight": 0 for key in ("co2", "nox", "fuel", "pm")})
    with pytest.raises(ValueError, match="inconnu"):
        c.update("rules", {"queue_weight": 3})


def test_real_simulation_switch_and_rule_update_are_logged(monkeypatch):
    from flowmind.evaluation import ROOT, Recorder, run_experiment
    c = RuntimeControl("fixed_time")
    original = Recorder.sample
    def sample(self, values):
        original(self, values)
        if values["elapsed_s"] == 5:
            c.update("rules", {"co2_weight": 3})
        if values["elapsed_s"] == 14:
            c.update("rules", {"switch_margin": .2})
    monkeypatch.setattr(Recorder, "sample", sample)
    result = run_experiment("fixed_time", seconds=25, runtime_control=c)
    assert result["status"] == "completed"
    assert result["controller"] == "interactive"
    assert result["control_changes"] == 2
    directory = ROOT / "runs" / result["run_id"]
    with (directory / "timeseries.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [float(row["elapsed_s"]) for row in rows] == list(range(1, 26))
    assert {row["controller"] for row in rows} == {"fixed_time", "rules"}
    assert {row["control_revision"] for row in rows} == {"0", "1", "2"}
    assert len(json.loads((directory / "control_events.json").read_text(encoding="utf-8"))["events"]) == 3
