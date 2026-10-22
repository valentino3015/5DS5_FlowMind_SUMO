"""Optional local LLM (Ollama) that turns retrieved log facts into a natural answer.

Owner: Adem.

Retrieval-augmented generation (RAG) in its simplest form:
1. retrieval: xai.explain.find_rows() picks the decisions the question is about
2. augmentation: those decisions become compact "field = value" FACTS in the prompt
3. generation: a small local model (config.yaml xai.llm_model, e.g. qwen2.5:3b)
   answers ONLY from those facts and cites them like [J2 t=160]

Prompt engineering: a fixed role, strict rules and one worked example (few-shot),
temperature 0. Guardrails after generation:
- grounding check: every number in the answer must appear in the facts or the
  question; otherwise the answer is rejected (probably invented) and the caller
  uses the rule-based template answer
- missing citations are added from the facts
If Ollama is not running, ask() returns None and the template answer is used.
Only the Python standard library is used (urllib), no new package.
The LLM never controls a light; it only rewords what the log already says.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

SYSTEM = (
    "You explain the decisions of an AI traffic-light controller to a non-expert. "
    "Each junction has specialist agents (emergency, incident, pedestrian, bus, flow); "
    "an orchestrator picks one by priority and a safety shield checks every decision.\n"
    "Rules:\n"
    "1. Use ONLY the FACTS. Never invent numbers, vehicles or reasons.\n"
    "2. End every sentence with the citation of the fact it uses, exactly like [J2 t=160].\n"
    "3. Answer in at most 4 short sentences, in plain English.\n"
    "4. Warnings come from the corridor coordinator BEFORE the decision: they are inputs that "
    "explain the decision, never results of it.\n"
    "5. Only if NO fact is about the question, answer exactly: This is not in the log."
)

EXAMPLE = (
    "Example:\n"
    "FACTS:\n"
    "- [J1 t=300] in control = bus; because = bus_we.4 is 95 s late, 80 m away; green phase given = 0; "
    "action = green for bus_we.4; other agents that asked = flow (lost: lower priority); "
    "fixed timer would choose = phase 2; safety shield = approved\n"
    "QUESTION: why did J1 give green to phase 0 at 300?\n"
    "ANSWER: At 300 s, J1 gave green to phase 0 because the bus agent took control: bus bus_we.4 was "
    "95 s late and 80 m away [J1 t=300]. The flow agent also asked but has a lower priority [J1 t=300]. "
    "A fixed timer would have chosen phase 2, and the safety shield approved the decision [J1 t=300].\n"
)


def build_prompt(question: str, facts: list[str]) -> str:
    return (f"{SYSTEM}\n\n{EXAMPLE}\nNow answer.\nFACTS:\n" + "\n".join(f"- {f}" for f in facts)
            + f"\nQUESTION: {question}\nANSWER:")


def ask(question: str, facts: list[str], url: str, model: str, timeout_s: float) -> str | None:
    """The model's answer, or None if Ollama is unreachable, too slow or answers nothing."""
    body = json.dumps({
        "model": model,
        "prompt": build_prompt(question, facts),
        "stream": False,
        "options": {"temperature": 0},
    }).encode("utf-8")
    request = urllib.request.Request(url.rstrip("/") + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            answer = json.loads(response.read().decode("utf-8")).get("response", "").strip()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return answer or None


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def is_grounded(answer: str, facts: list[str], question: str) -> bool:
    """True if every number in the answer also appears in the facts or the question."""
    allowed = numbers(" ".join(facts) + " " + question)
    return numbers(answer) <= allowed


def is_consistent(answer: str, agents_in_control: set[str]) -> bool:
    """True if every agent the answer says "took control" / "was in control" really was in control
    in one of the retrieved decisions (catches answers that swap the winner and the loser)."""
    said = re.findall(r"(\w+) agent (?:took|was in|had|kept|maintained) control|(\w+) was in control", answer.lower())
    names = {a or b for a, b in said}
    return names <= {a.lower() for a in agents_in_control}


def tidy(answer: str) -> str:
    """Drop a trailing "This is not in the log." when the model also gave a real, cited answer."""
    if "[" in answer and answer.rstrip().endswith("This is not in the log."):
        return answer.rstrip()[: -len("This is not in the log.")].rstrip()
    return answer


def ensure_citations(answer: str, citations: list[str]) -> str:
    """If the model gave no [J.. t=..] citation, list the facts it was given as sources."""
    if any(c in answer for c in citations):
        return answer
    return answer + "\nSources: " + " ".join(citations)
