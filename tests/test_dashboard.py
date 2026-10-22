"""Exercise local dashboard HTTP controls without launching a simulator."""
import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def load_dashboard():
    path = Path(__file__).resolve().parent.parent / "scripts/dashboard.py"
    spec = importlib.util.spec_from_file_location("flowmind_series_dashboard", path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def test_series_units_partial_rows_and_bounded_history(tmp_path, monkeypatch):
    app = load_dashboard()
    monkeypatch.setattr(app, "ROOT", tmp_path)
    directory = tmp_path / "runs" / "trial"
    directory.mkdir(parents=True)
    path = directory / "timeseries.csv"
    header = "elapsed_s,vehicles,halting,queue_mean_lane,speed_sum_m_s,co2_mg_s\n"
    path.write_text(header + "1,2,1,0.5,10,1000000\n2,0,0,0,0,0\n3,2,", encoding="utf-8-sig")
    data = app.read_series("trial")
    assert data["sample_count"] == 2
    assert data["points"][0]["speed_kmh"] == 18
    assert data["points"][0]["co2_g_s"] == 1000
    assert data["points"][1]["speed_kmh"] is None
    path.write_text(header + "".join(f"{i},2,1,0.5,10,1000\n" for i in range(1, 2001)), encoding="utf-8-sig")
    data = app.read_series("trial")
    assert data["sample_count"] == 2000
    assert len(data["points"]) == 600
    assert data["points"][0]["elapsed_s"] == 1
    assert data["points"][-1]["elapsed_s"] == 2000
    data = app.read_series("trial", 1400, 1410)
    assert data["sample_count"] == 11
    assert data["points"][0]["elapsed_s"] == 1400
    assert data["points"][-1]["elapsed_s"] == 1410
    import pytest
    with pytest.raises(ValueError):
        app.read_series("../outside")


def test_new_run_does_not_reuse_previous_live_data(tmp_path, monkeypatch):
    import os
    app = load_dashboard()
    monkeypatch.setattr(app, "ROOT", tmp_path)
    old = tmp_path / "runs" / "old"
    old.mkdir(parents=True)
    (old / "metadata.json").write_text('{"run_id":"old"}')
    (old / "live.json").write_text('{"elapsed_s":600}')
    os.utime(old / "metadata.json", (100, 100))
    assert app.latest_live(200) is None
    new = tmp_path / "runs" / "new"
    new.mkdir()
    (new / "metadata.json").write_text('{"run_id":"new","status":"running"}')
    assert app.latest_live(200) == {"run_id": "new", "status": "running"}


def test_dashboard_rejects_concurrent_and_cross_origin_controls(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parent.parent / "scripts/dashboard.py"
    spec = importlib.util.spec_from_file_location("flowmind_test_dashboard", path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    release = threading.Event()
    def fake_worker(options):
        release.wait(5)
    monkeypatch.setattr(app, "worker", fake_worker)
    server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def post(route, body, origin=None):
        headers = {"X-FlowMind": "dashboard", "Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        request = Request(base + route, data=json.dumps(body).encode(), headers=headers)
        try:
            response = urlopen(request, timeout=5)
            return response.status
        except HTTPError as exc:
            return exc.code
    try:
        options = dict(controllers=["fixed_time"], scenario="3x3grid", seconds=60, seeds=[100], gui=False)
        assert post("/api/start", options) == 200
        assert post("/api/start", options) == 409
        assert post("/api/stop", {}, "https://example.com") == 403
        assert not app.stop.is_set()
        assert post("/api/stop", {}) == 200
        assert app.stop.is_set()
        assert post("/api/start", options | {"seconds": -1}) == 400
        assert urlopen(base, timeout=5).status == 200
        assert urlopen(base + "/api/state", timeout=5).status == 200
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "best.zip").write_bytes(b"unused by request validation")
        (model_dir / "manifest.json").write_text(json.dumps({
            "scenario":"3x3grid", "train_seeds":[0], "validation_seeds":[1000],
            "safety":{"min_green":10, "max_green":35, "max_red":30, "blocked_occupancy":.8}}))
        monkeypatch.setattr(app, "MODEL_DIR", model_dir)
        monkeypatch.setattr(app, "latest_live", lambda *args: {"scenario":"3x3grid", "seed":10020})
        app.runtime_control = app.RuntimeControl()
        before = app.runtime_control.snapshot()
        assert post("/api/control", {"method":"emission_model", "rules":{"min_green":50}}) == 400
        assert app.runtime_control.snapshot() == before
        assert post("/api/control", {"method":"emission_model", "rules":{"min_green":10}}) == 200
    finally:
        release.set()
        server.shutdown()
        server.server_close()
