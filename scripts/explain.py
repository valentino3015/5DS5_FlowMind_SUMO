"""XAI: explain what the orchestrator did in a run.

Owner: Adem.

    python scripts/explain.py RUN_DIR --story                     timeline of the important moments
    python scripts/explain.py RUN_DIR --at J2 160                 one decision: why, why not, what if
    python scripts/explain.py RUN_DIR --ask "why did J2 switch at 160?"           template answer
    python scripts/explain.py RUN_DIR --ask "why did J2 switch at 160?" --llm     local LLM (Ollama)
    python scripts/explain.py RUN_DIR --sensitivity J2 160        which inputs mattered
                                                                  (needs run.py --save-states)

RUN_DIR is a folder written by scripts/run.py --controller orchestrated.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xai import llm  # noqa: E402
from xai.explain import cite, explain_decision, fact_line, find_rows, load_run, run_story, template_answer  # noqa: E402
from xai.sensitivity import analyze, load_state, sensitivity_text  # noqa: E402


def make_orchestrator_function():
    """scripts/run.py's make_orchestrator (same agents as the real run)."""
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_orchestrator


def main():
    p = argparse.ArgumentParser(description="Explain the orchestrator's decisions in a run.")
    p.add_argument("run_dir", help="folder written by scripts/run.py")
    p.add_argument("--story", action="store_true", help="timeline of the important moments")
    p.add_argument("--at", nargs=2, metavar=("TLS", "TIME"), help="explain one decision, e.g. --at J2 160")
    p.add_argument("--ask", metavar="QUESTION", help="answer a question about the run")
    p.add_argument("--llm", action="store_true", help="use the local LLM (Ollama) to word the answer")
    p.add_argument("--sensitivity", nargs=2, metavar=("TLS", "TIME"), help="which inputs mattered")
    args = p.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    priority = config["orchestrator"]["priority"]
    agents = priority + ["operator", "max_pressure"]
    xcfg = config["xai"]
    rows = load_run(args.run_dir)

    if args.story:
        print("Story of the run:")
        print("\n".join(run_story(rows)))

    if args.at:
        found = find_rows(rows, f"{args.at[0]} {args.at[1]}", agents, max_rows=1)
        print(explain_decision(found[0], priority) if found else "No decision found there.")

    if args.ask:
        found = find_rows(rows, args.ask, agents, xcfg["max_rows"])
        answer, mode = None, "template (rule-based, no LLM)"
        if args.llm and found:
            facts = [fact_line(r, priority) for r in found]
            answer = llm.ask(args.ask, facts, xcfg["llm_url"], xcfg["llm_model"], xcfg["llm_timeout_s"])
            if answer is None:
                mode = "template (the local LLM is not reachable, so the rule-based answer is used)"
            elif not llm.is_grounded(answer, facts, args.ask):
                print(f"(LLM answer rejected by the grounding check, it contained numbers not in the log:\n"
                      f" {answer})\n")
                answer, mode = None, "template (the LLM answer failed the grounding check)"
            elif not llm.is_consistent(answer, {r["agent"] for r in found}):
                print(f"(LLM answer rejected by the consistency check, it named the wrong agent in control:\n"
                      f" {answer})\n")
                answer, mode = None, "template (the LLM answer failed the consistency check)"
            else:
                answer = llm.ensure_citations(llm.tidy(answer), [cite(r) for r in found])
                mode = f"local LLM {xcfg['llm_model']} (answers only from {len(found)} retrieved log decisions)"
        print(f"Q: {args.ask}\nAnswered by: {mode}\n")
        print(answer or template_answer(args.ask, found, priority))

    if args.sensitivity:
        tls, time = args.sensitivity[0].upper(), float(args.sensitivity[1])
        saved = load_state(Path(args.run_dir), tls, time)
        if saved is None:
            print("No states.jsonl in this run: run scripts/run.py again with --save-states.")
        else:
            state, events = saved
            result = analyze(state, events, make_orchestrator_function(), config, xcfg["perturb_cars"])
            logged = next((r for r in rows if r["tls_id"] == tls and r["time"] == state.time), None)
            print(sensitivity_text(result, tls, state.time, logged))

    if not (args.story or args.at or args.ask or args.sensitivity):
        p.print_help()


if __name__ == "__main__":
    main()
