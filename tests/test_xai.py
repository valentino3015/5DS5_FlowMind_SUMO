"""Tests for the XAI layer (xai/): explanations, retrieval, LLM guardrails, sensitivity.

Owner: Adem. No SUMO and no Ollama needed.
"""
import importlib.util
from dataclasses import replace
from pathlib import Path

import yaml

from flowmind.contracts import Claim, SpecialVehicle, make_example_state
from flowmind.control.orchestrator import Orchestrator
from xai import llm
from xai.explain import explain_decision, find_rows, parse_claims, parse_question, what_if, why_not
from xai.sensitivity import analyze, state_from_dict

ROOT = Path(__file__).parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
PRIORITY = CONFIG["orchestrator"]["priority"]
AGENTS = PRIORITY + ["operator", "max_pressure"]


def row(tls="J2", time=160.0, agent="emergency", reason="ambulance amb_1 50 m away", final=0,
        claims="emergency:1.00:ambulance amb_1 50 m away; flow:0.10:normal traffic", would="2",
        events="emergency amb_1 eta 4s", note="", explanation="green for ambulance amb_1"):
    return {"tls_id": tls, "time": time, "agent": agent, "reason": reason, "final": final, "proposed": final,
            "explanation": explanation, "claims": claims, "fixed_time_would": would,
            "events": events, "shield_note": note}


# --- the orchestrator records every claim ---
class Asker:
    def __init__(self, name, claim):
        self.name, self.claim = name, claim

    def wants_control(self, state, events):
        return self.claim

    def act(self, state, events, allowed):
        return 0

    def explain(self):
        return "test"


def test_decision_lists_all_claims_most_important_first():
    agents = [Asker("flow", Claim(0.1, "normal traffic")), Asker("emergency", Claim(1.0, "ambulance"))]
    d = Orchestrator(agents, PRIORITY, 10, fallback=Asker("max_pressure", None)).decide(
        make_example_state(), [], [True] * 4)
    assert d.agent == "emergency"
    assert parse_claims(d.claims) == [("emergency", 1.0, "ambulance"), ("flow", 0.1, "normal traffic")]


# --- contrastive and counterfactual explanations ---
def test_why_not_priority():
    (text,) = why_not(row(), PRIORITY)
    assert "flow also asked" in text and "lower priority" in text


def test_why_not_min_hold_operator_and_nobody():
    held = row(agent="bus", reason="keeps control for min_hold (5/10 s)",
               claims="emergency:1.00:ambulance; bus:0.50:late bus")
    assert "minimum hold time" in why_not(held, PRIORITY)[0]
    assert "operator override" in why_not(row(agent="operator", claims=""), PRIORITY)[0]
    assert "No specialist asked" in why_not(row(agent="max_pressure", claims=""), PRIORITY)[0]


def test_what_if_and_full_paragraph():
    assert what_if(row(would="2")) == "A fixed timer would have chosen phase 2 instead of phase 0."
    assert what_if(row(would="0")) == "A fixed timer would have chosen the same phase."
    text = explain_decision(row(note="phase 3 not allowed now"), PRIORITY)
    assert "J2" in text and "emergency" in text and "safety shield changed" in text


# --- retrieval ---
def test_question_parsing_ignores_phase_numbers():
    q = parse_question("why did J2 choose phase 0 at 160 s for amb_1?", AGENTS)
    assert q["junctions"] == ["J2"] and q["times"] == [160.0] and q["vehicles"] == ["amb_1"]


def test_find_rows_nearest_time_and_vehicle():
    rows = [row(time=t) for t in (150.0, 155.0, 161.0)] + [row(tls="J1", time=160.0, reason="normal",
                                                               agent="flow", events="",
                                                               explanation="phase 2 has the highest queue")]
    assert [r["time"] for r in find_rows(rows, "J2 at 160", AGENTS)] == [161.0]
    assert {r["tls_id"] for r in find_rows(rows, "what about amb_1", AGENTS)} == {"J2"}
    assert find_rows(rows, "what is the weather?", AGENTS) == []


# --- LLM guardrails (no Ollama needed) ---
def test_llm_unreachable_returns_none():
    assert llm.ask("why?", ["[J2 t=160] x"], "http://127.0.0.1:9", "qwen2.5:3b", timeout_s=1) is None


def test_prompt_contains_facts_and_rules():
    prompt = llm.build_prompt("why did J2 switch?", ["[J2 t=160] in control = emergency"])
    assert "[J2 t=160] in control = emergency" in prompt and "Use ONLY the FACTS" in prompt
    assert prompt.rstrip().endswith("ANSWER:")


def test_grounding_and_consistency_checks():
    facts = ["[J2 t=160] in control = emergency; because = ambulance amb_1 50 m away"]
    assert llm.is_grounded("The ambulance was 50 m away [J2 t=160].", facts, "J2 at 160")
    assert not llm.is_grounded("The ambulance was 80 m away [J2 t=160].", facts, "J2 at 160")
    assert llm.is_consistent("The emergency agent took control [J2 t=160].", {"emergency"})
    assert not llm.is_consistent("The flow agent took control [J2 t=160].", {"emergency"})
    assert llm.tidy("Answer [J2 t=160]. This is not in the log.") == "Answer [J2 t=160]."
    assert llm.ensure_citations("No citation here.", ["[J2 t=160]"]).endswith("Sources: [J2 t=160]")


# --- sensitivity ---
def test_removing_the_ambulance_changes_the_decision():
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    amb = SpecialVehicle("amb_1", "emergency", "west_in_0", distance=80.0, speed=12.0, delay=None)
    s = make_example_state()
    s = replace(s, special_vehicles=s.special_vehicles + (amb,), time_in_phase=20.0)
    result = analyze(s, [], run.make_orchestrator, CONFIG)
    assert result["agent"] == "emergency"
    assert any(c.startswith("without amb_1 on the approach:") for c in result["changes"])


def test_state_roundtrip_from_json():
    from dataclasses import asdict
    s = make_example_state()
    assert state_from_dict(asdict(s)) == s


def test_warning_is_replayed_and_can_be_removed():
    """A coordinator warning alone makes the emergency agent take control; removing it changes that."""
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    from flowmind.contracts import Event
    s = replace(make_example_state(), time_in_phase=20.0)
    warning = Event("emergency", "J1", s.time + 30, {"vehicle_id": "amb_9", "eta": 10.0, "lane_id": "east_in_0"})
    result = analyze(s, [warning], run.make_orchestrator, CONFIG)
    assert result["agent"] == "emergency"
    assert any("without the coordinator's warning about amb_9" in c for c in result["changes"])


# --- xai/service.py: the entry point shared by scripts/explain.py and the dashboard ---
def write_log(folder, rows):
    import csv
    from flowmind.logger import COLUMNS
    keys = {"time": "time", "tls_id": "tls_id", "agent": "agent", "urgency": None, "reason": "reason",
            "proposed": "proposed", "final": "final", "shield_note": "shield_note", "explanation": "explanation",
            "events": "events", "latency_ms": None, "fixed_time_would": "fixed_time_would", "claims": "claims"}
    with open(folder / "decisions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r[keys[c]] if keys.get(c) else 0 for c in COLUMNS])


def test_service_story_explain_and_template_answer(tmp_path):
    from xai import service
    write_log(tmp_path, [row(time=155.0, agent="flow", reason="normal traffic", claims="flow:0.10:normal traffic",
                             events="", explanation="heuristic"), row()])
    assert any("emergency took control" in line for line in service.story(tmp_path))
    assert "emergency was in control" in service.explain_at(tmp_path, "J2", 160, CONFIG)
    result = service.answer(tmp_path, "why did J2 give green at 160?", CONFIG, use_llm=False)
    assert result["mode"].startswith("template") and "[J2 t=160]" in result["answer"] and not result["rejected"]


def test_service_sensitivity_needs_saved_states(tmp_path):
    from xai import service
    write_log(tmp_path, [row()])
    assert "--save-states" in service.sensitivity(tmp_path, "J2", 160, CONFIG, make_orchestrator=None)
