"""One entry point for the XAI layer, shared by scripts/explain.py and the dashboard.

Owner: Adem.

Every function reads a run's decision log (the folder holding decisions.csv) and returns text.
Nothing here controls a light. The local LLM is optional: without it the rule-based template
answers, and an LLM answer is only kept when it passes the grounding and consistency checks.
"""
from __future__ import annotations

from pathlib import Path

from xai import llm
from xai.explain import cite, explain_decision, fact_line, find_rows, load_run, run_story, template_answer
from xai.sensitivity import analyze, load_state, sensitivity_text


def agents_of(config: dict) -> list[str]:
    return config["orchestrator"]["priority"] + ["operator", "max_pressure"]


def story(run_dir: Path) -> list[str]:
    return run_story(load_run(run_dir))


def explain_at(run_dir: Path, tls: str, time: float, config: dict) -> str:
    found = find_rows(load_run(run_dir), f"{tls} {time:g}", agents_of(config), max_rows=1)
    return explain_decision(found[0], config["orchestrator"]["priority"]) if found else "No decision found there."


def answer(run_dir: Path, question: str, config: dict, use_llm: bool = False) -> dict:
    """{"answer", "mode", "rejected"}: rejected holds an LLM answer that failed a check (else "")."""
    priority, xcfg = config["orchestrator"]["priority"], config["xai"]
    found = find_rows(load_run(run_dir), question, agents_of(config), xcfg["max_rows"])
    result = {"answer": None, "mode": "template (rule-based, no LLM)", "rejected": ""}
    if use_llm and found:
        facts = [fact_line(r, priority) for r in found]
        text = llm.ask(question, facts, xcfg["llm_url"], xcfg["llm_model"], xcfg["llm_timeout_s"])
        if text is None:
            result["mode"] = "template (the local LLM is not reachable, so the rule-based answer is used)"
        elif not llm.is_grounded(text, facts, question):
            result.update(mode="template (the LLM answer failed the grounding check)",
                          rejected=f"it contained numbers not in the log: {text}")
        elif not llm.is_consistent(text, {r["agent"] for r in found}):
            result.update(mode="template (the LLM answer failed the consistency check)",
                          rejected=f"it named the wrong agent in control: {text}")
        else:
            result.update(answer=llm.ensure_citations(llm.tidy(text), [cite(r) for r in found]),
                          mode=f"local LLM {xcfg['llm_model']} (answers only from {len(found)} retrieved log decisions)")
    if result["answer"] is None:
        result["answer"] = template_answer(question, found, priority)
    return result


def sensitivity(run_dir: Path, tls: str, time: float, config: dict, make_orchestrator) -> str:
    saved = load_state(Path(run_dir), tls, time)
    if saved is None:
        return "No saved junction states in this run (states.jsonl): run it again with --save-states."
    state, events = saved
    result = analyze(state, events, make_orchestrator, config, config["xai"]["perturb_cars"])
    logged = next((r for r in load_run(run_dir) if r["tls_id"] == tls and r["time"] == state.time), None)
    return sensitivity_text(result, tls, state.time, logged)
