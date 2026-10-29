"""Check metric units, unfinished trips, and fair pairing."""
import pytest
from flowmind.kpis import compare_rows, compute_kpis, write_csv
from flowmind.evaluation import run_experiment


def test_metrics_integrate_seconds_and_keep_unfinished_trips(tmp_path):
    row = dict(elapsed_s=1, vehicles=2, departed=2, arrived=0, pending=0,
               halting=1, queue_mean_lane=.5, queue_max_lane=1, speed_sum_m_s=10,
               co2_mg_s=1_000_000, nox_mg_s=1000, pmx_mg_s=10,
               fuel_mg_s=500_000, collisions=0, teleports=0)
    write_csv(tmp_path / "timeseries.csv", [row, row | {"elapsed_s": 2, "departed": 0, "arrived": 1}])
    emission = '<emissions CO2_abs="1000000" NOx_abs="1000" PMx_abs="10" fuel_abs="500000"/>'
    (tmp_path / "tripinfo.xml").write_text(f'<tripinfos><tripinfo arrival="2" waitingTime="2" duration="2">{emission}</tripinfo><tripinfo arrival="-1" waitingTime="6" duration="1">{emission}</tripinfo></tripinfos>')
    r = compute_kpis(tmp_path)
    assert r["mean_wait_s"] == 4
    assert r["mean_completed_duration_s"] == 2
    assert r["unfinished"] == 1
    assert r["co2_kg"] == 2
    assert r["co2_kg_per_departed"] == 1
    assert r["nox_g"] == 2
    assert r["fuel_kg"] == 1
    assert r["mean_speed_kmh"] == 18


def trial(**changes):
    return dict(status="completed", controller="fixed_time", scenario="3x3grid",
                situation="trafic_fourni", seconds=60, seed=100, input_hash="same",
                protocol_hash="same", model_hash="none", mean_wait_s=50,
                arrived=10) | changes


def test_pairing_rejects_different_conditions_and_stopped_runs():
    rows = [trial(), trial(controller="flow_heuristic", mean_wait_s=35, arrived=12),
            trial(controller="flow_heuristic", seed=101, mean_wait_s=100),
            trial(controller="flow_heuristic", protocol_hash="different"),
            trial(controller="flow_heuristic", status="stopped")]
    result = compare_rows(rows)
    wait = next(r for r in result if r["metric"] == "mean_wait_s")
    assert wait["paired_seeds"] == 1
    assert wait["improvement_pct"] == 30
    assert next(r for r in result if r["metric"] == "arrived")["improvement_pct"] == 20


def test_zero_reference_has_no_percentage():
    rows = [trial(mean_wait_s=0), trial(controller="flow_heuristic", mean_wait_s=0)]
    assert compare_rows(rows)[0]["change_pct"] is None


def test_atomic_json_retries_windows_sharing_violation(tmp_path, monkeypatch):
    from pathlib import Path
    from flowmind.kpis import atomic_json
    original = Path.replace
    attempts = []
    def transient(self, target):
        attempts.append(target)
        if len(attempts) == 1:
            raise PermissionError("Temporary Windows sharing violation")
        return original(self, target)
    monkeypatch.setattr(Path, "replace", transient)
    atomic_json(tmp_path / "live.json", {"value": 42})
    assert len(attempts) == 2
    assert '42' in (tmp_path / "live.json").read_text()


def test_missing_model_cannot_be_passed_off_as_trained():
    from flowmind.env import ROOT
    if (ROOT / "runs/models/flow_ppo.zip").exists():
        pytest.skip("A model has since been added")
    with pytest.raises(ValueError, match="modèle"):
        run_experiment("flow_model", seconds=10)


def test_stopped_run_closes_sumo_and_preserves_partial_metrics():
    import threading
    stop = threading.Event()
    stop.set()
    result = run_experiment("fixed_time", seconds=10, stop_event=stop)
    assert result["status"] == "stopped"
    assert result["actual_seconds"] == 1
