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

from xai import service  # noqa: E402


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
    run_dir = Path(args.run_dir)

    if args.story:
        print("Story of the run:")
        print("\n".join(service.story(run_dir)))

    if args.at:
        print(service.explain_at(run_dir, args.at[0].upper(), float(args.at[1]), config))

    if args.ask:
        result = service.answer(run_dir, args.ask, config, use_llm=args.llm)
        if result["rejected"]:
            print(f"(LLM answer rejected: {result['rejected']})\n")
        print(f"Q: {args.ask}\nAnswered by: {result['mode']}\n")
        print(result["answer"])

    if args.sensitivity:
        tls, time = args.sensitivity[0].upper(), float(args.sensitivity[1])
        print(service.sensitivity(run_dir, tls, time, config, make_orchestrator_function()))

    if not (args.story or args.at or args.ask or args.sensitivity):
        p.print_help()


if __name__ == "__main__":
    main()
