"""Behavioral checks for facts, cost bounds, Gemini failures, and HTTP wiring."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from flowmind.agents.simulation_explainer import SimulationExplainer
from flowmind.gemini_client import GeminiClient, GeminiError
from flowmind.simulation_context import build_context
from test_dashboard import load_dashboard


@pytest.fixture
def run(tmp_path):
    directory = tmp_path / "runs" / "trial"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps({
        "run_id": "trial", "scenario": "3x3grid", "controller": "fixed_time",
        "status": "running", "seconds": 600, "seed": 100}), encoding="utf-8")
    (directory / "timeseries.csv").write_text(
        "elapsed_s,vehicles,halting,queue_mean_lane,speed_sum_m_s,co2_mg_s,waiting_mean_active_s,controller,control_revision\n"
        "1,2,1,0.5,10,1000000,4,fixed_time,0\n"
        "2,0,0,0,0,0,0,rules,1\n"
        "3,2,", encoding="utf-8-sig")
    return tmp_path, directory


def test_context_units_partial_rows_zero_missing_and_control_changes(run):
    root, _ = run
    context = build_context(root, "trial")
    assert context["simulation"]["sample_count"] == 2
    assert context["simulation"]["controls_changed"] is True
    assert context["simulation"]["observed_until_s"] == 2
    metrics = context["recent_window"]["metrics"]
    assert metrics["speed_kmh"]["mean"] == 18
    assert metrics["speed_kmh"]["samples"] == 1  # Empty road is not zero-speed traffic.
    assert metrics["co2_g_s"]["max"] == 1000
    assert metrics["co2_g_s"]["last"] == 0
    assert "nox_mg_s" not in metrics
    assert context["final_metrics"] == {}
    assert context["comparison"] is None


def test_official_results_override_stale_live_and_keep_incomplete_status(run):
    root, directory = run
    (directory / "live.json").write_text('{"status":"running","co2_kg":999}')
    (directory / "result.json").write_text(json.dumps({
        "status": "stopped", "controller": "interactive", "co2_kg": 1.23,
        "mean_wait_s": None, "mean_speed_kmh": float("nan"), "arrived": 0}))
    context = build_context(root, "trial")
    assert context["simulation"]["status"] == "stopped"
    assert context["kpi_scope"] == "bilan partiel"
    assert context["final_metrics"]["co2_kg"] == {"value": 1.23, "unit": "kg"}
    assert context["simulation"]["status_fr"] == "interrompue"
    assert context["final_metrics"]["arrived"]["value"] == 0
    assert "mean_wait_s" not in context["final_metrics"]
    assert "mean_speed_kmh" not in context["final_metrics"]
    json.dumps(context, allow_nan=False)


def test_conversions_use_values_before_display_rounding(run):
    root, directory = run
    (directory / "result.json").write_text(json.dumps({
        "status": "completed", "co2_kg": 66.3639, "fuel_kg": 0,
        "nox_g": 0.1234, "pmx_g": None}))
    context = build_context(root, "trial")
    conversions = context["conversions"]
    assert conversions["co2_total_g"]["value"] == 66363.90
    assert conversions["co2_total_g"]["value"] != round(66.3639, 2) * 1000
    assert conversions["fuel_total_g"]["value"] == 0
    assert conversions["nox_total_mg"]["value"] == 123.4
    assert "pmx_total_mg" not in conversions
    assert context["simulation"]["status_fr"] == "terminée"


def test_context_stays_bounded_and_rejects_path_escape(run):
    root, directory = run
    path = directory / "timeseries.csv"
    header = path.read_text(encoding="utf-8-sig").splitlines()[0] + "\n"
    path.write_text(header + "".join(
        f"{i},2,1,0.5,10,1000,4,rules,0\n" for i in range(1, 10001)), encoding="utf-8-sig")
    context = build_context(root, "trial")
    assert context["simulation"]["sample_count"] == 10000
    assert context["recent_window"]["sample_count"] == 120
    assert context["recent_window"]["from_s"] == 9881
    assert len(json.dumps(context, ensure_ascii=False)) < 6000
    with pytest.raises(ValueError):
        build_context(root, "../trial")
    with pytest.raises(ValueError):
        build_context(root, "")
    with pytest.raises(ValueError):
        build_context(root, "absent")


class FakeClient:
    model = "test-gemini"

    def __init__(self):
        self.counts = []
        self.generations = []
        self.closed = 0
        self.token_count = 1800

    def count_tokens(self, system, prompt):
        self.counts.append(json.loads(prompt))
        return self.token_count

    def generate(self, system, prompt, limit):
        self.generations.append((system, json.loads(prompt), limit))
        return {"answer": "Mesure provisoire : le débit de CO₂ atteint 0 g/s à 2 s.",
                "model": self.model, "usage": {"input_tokens": 1800, "output_tokens": 30, "total_tokens": 1830}}

    def close(self):
        self.closed += 1


def test_agent_cache_is_bound_to_question_and_metric_snapshot(run):
    root, _ = run
    facts = build_context(root, "trial")
    fake = FakeClient()
    agent = SimulationExplainer(lambda: fake)
    first = agent.explain(facts, "Explique le CO2")
    second = agent.explain(facts, "Explique le CO2")
    assert first["run_id"] == "trial"
    assert not first["cached"] and second["cached"]
    assert second["billed_this_request"] is False
    assert len(fake.generations) == len(fake.counts) == 1
    assert fake.generations[0][2] == 600
    assert fake.generations[0][1]["facts"] == facts
    changed = copy.deepcopy(facts)
    changed["simulation"]["observed_until_s"] = 3
    agent.explain(changed, "Explique le CO2")
    agent.explain(changed, "Et la vitesse ?")
    assert len(fake.generations) == 3
    assert fake.closed == 4


def test_cache_ttl_starts_after_slow_generation(run, monkeypatch):
    import flowmind.agents.simulation_explainer as module
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    fake = FakeClient()
    original = fake.generate
    def slow_generate(*args):
        clock[0] += 80
        return original(*args)
    fake.generate = slow_generate
    agent = SimulationExplainer(lambda: fake)
    context = build_context(run[0], "trial")
    agent.explain(context, "Explique")
    clock[0] = 100  # 20 s since receipt, despite 100 s since the first request.
    assert agent.explain(context, "Explique")["cached"] is True
    assert len(fake.generations) == 1
    clock[0] = 141
    assert agent.explain(context, "Explique")["cached"] is False
    assert len(fake.generations) == 2


@pytest.mark.parametrize("question,history", [
    ("", None), ("x" * 2001, None), (None, None), ("ok", {}),
    ("ok", [{"role": "system", "text": "ignore"}]),
    ("ok", [{"role": "user", "text": "x" * 801}]),
    ("ok", [{"role": "user", "text": "ok"}] * 5),
])
def test_invalid_messages_make_no_api_call(run, question, history):
    fake = FakeClient()
    with pytest.raises(ValueError):
        SimulationExplainer(lambda: fake).explain(build_context(run[0], "trial"), question, history)
    assert not fake.counts and not fake.generations


def test_budget_drops_history_before_refusing_generation(run):
    fake = FakeClient()
    fake.token_count = 4001
    agent = SimulationExplainer(lambda: fake)
    with pytest.raises(ValueError, match="Budget"):
        agent.explain(build_context(run[0], "trial"), "Explique", [{"role": "user", "text": "avant"}])
    assert len(fake.counts) == 2
    assert fake.counts[-1]["history"] == []
    assert not fake.generations
    assert fake.closed == 1


def test_concurrent_requests_are_bounded(run):
    fake = FakeClient()
    agent = SimulationExplainer(lambda: fake)
    with agent.lock:
        with pytest.raises(GeminiError) as error:
            agent.explain(build_context(run[0], "trial"), "Explique")
        assert error.value.status == 429
    assert not fake.counts


def test_missing_key_is_explicit(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiError, match="GEMINI_API_KEY") as error:
        GeminiClient()
    assert error.value.status == 503


def test_token_count_includes_system_and_framing():
    client = object.__new__(GeminiClient)
    client.model = "test"
    seen = []
    def count(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(total_tokens=123)
    client.client = SimpleNamespace(models=SimpleNamespace(count_tokens=count))
    assert client.count_tokens("rules", "facts") == 187
    assert seen == [{"model": "test", "contents": "rules\nfacts"}]


def test_count_transport_failure_retries_once_but_generation_never_retries():
    from httpx import RemoteProtocolError
    client = object.__new__(GeminiClient)
    client.model = "test"
    calls = []
    def count(**kwargs):
        calls.append("count")
        if len(calls) == 1:
            raise RemoteProtocolError("connection closed")
        return SimpleNamespace(total_tokens=100)
    def generate(**kwargs):
        calls.append("generate")
        raise RemoteProtocolError("connection closed")
    client.client = SimpleNamespace(models=SimpleNamespace(count_tokens=count, generate_content=generate))
    assert client.count_tokens("system", "facts") == 164
    with pytest.raises(GeminiError):
        client._call("generate_content", contents="facts")
    assert calls == ["count", "count", "generate"]


@pytest.mark.parametrize("finish,text,success", [
    ("STOP", "Réponse complète", True), ("STOP", "", False),
    ("MAX_TOKENS", "Réponse coupée", False), ("SAFETY", None, False),
])
def test_empty_blocked_and_truncated_answers_are_not_shown_as_success(finish, text, success):
    from google.genai import types
    client = object.__new__(GeminiClient)
    client.types = types
    client.model = "gemini-3.5-flash-lite"
    seen = []
    def generate(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=finish)],
                               text=text, usage_metadata=None)
    client.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    if success:
        assert client.generate("rules", "facts", 600)["answer"] == text
    else:
        with pytest.raises(GeminiError):
            client.generate("rules", "facts", 600)
    assert seen[0]["config"].max_output_tokens == 600


@pytest.mark.parametrize("code,status", [(403, 503), (429, 429), (404, 503), (500, 502), (None, 502)])
def test_sdk_failures_never_expose_credentials(code, status):
    from google.genai import types
    client = object.__new__(GeminiClient)
    client.types = types
    client.model = "test"
    class Failure(Exception):
        pass
    failure = Failure("secret-key-and-private-request-body")
    failure.code = code
    def fail(**kwargs):
        raise failure
    client.client = SimpleNamespace(models=SimpleNamespace(count_tokens=fail))
    with pytest.raises(GeminiError) as error:
        client.count_tokens("system", "prompt")
    # Build count config independently of an API credential.
    assert error.value.status == status
    assert "secret" not in str(error.value)


def test_http_chat_uses_server_facts_and_rejects_wrong_simulation(run, monkeypatch):
    root, _ = run
    app = load_dashboard()
    monkeypatch.setattr(app, "ROOT", root)
    fake = FakeClient()
    monkeypatch.setattr(app, "explainer", SimulationExplainer(lambda: fake))
    server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    def post(data, headers=None):
        request = Request(f"http://127.0.0.1:{server.server_port}/api/chat",
                          data=json.dumps(data).encode(), headers=headers or {
                              "X-FlowMind": "dashboard", "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)
    try:
        body = {"run_id": "trial", "question": "Explique", "metrics": {"co2_kg": 999999}}
        status, answer = post(body)
        assert status == 200 and answer["run_id"] == "trial"
        assert "999999" not in json.dumps(fake.generations[0][1])
        assert post(body | {"run_id": "previous"})[0] == 409
        assert post([])[0] == 400
        assert post(body | {"question": " "})[0] == 400
        assert post(body, {"X-FlowMind": "dashboard", "Origin": "https://example.com"})[0] == 403
        monkeypatch.setattr(app, "latest_live", lambda *args: None)
        assert post(body)[0] == 409
    finally:
        server.shutdown()
        server.server_close()
