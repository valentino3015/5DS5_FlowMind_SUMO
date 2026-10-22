"""Explanations of the orchestrator's decisions, written from the decision log. No LLM here.

Owner: Adem.

Techniques (all rule-based, so they are exact and reproducible):
- contrastive "why X and not Y": every agent that asked is in the log (column
  claims), so we can say who lost and why (priority, urgency, min_hold, override)
- counterfactual "what if": the column fixed_time_would says what a plain fixed
  timer would have done in the same situation
- retrieval: find_rows() picks the log rows a question is about (junction, time,
  vehicle, agent); these rows are the only facts given to the optional LLM
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

DEFAULT_AGENTS = ("max_pressure", "fixed_time", "green_wave", "flow")  # normal-traffic controllers


def load_run(run_dir: Path) -> list[dict]:
    """Rows of decisions.csv, with time and phases as numbers."""
    with open(Path(run_dir) / "decisions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["time"] = float(row["time"])
        row["final"] = int(row["final"])
        row["proposed"] = int(row["proposed"])
    return rows


def parse_claims(text: str) -> list[tuple[str, float, str]]:
    """"emergency:1.00:ambulance ...; flow:0.10:normal" -> [("emergency", 1.0, "ambulance ..."), ...]."""
    claims = []
    for part in (text or "").split("; "):
        if part.count(":") >= 2:
            name, urgency, reason = part.split(":", 2)
            claims.append((name, float(urgency), reason))
    return claims


def why_not(row: dict, priority: list[str]) -> list[str]:
    """Contrastive part: why the other agents that asked did not get control."""
    agent, claims = row["agent"], parse_claims(row.get("claims", ""))
    rank = {name: i for i, name in enumerate(priority)}
    others = [c for c in claims if c[0] != agent]
    if agent == "operator":
        return ["An operator override was active: it beats every agent (the safety shield still checks it)."]
    if row["reason"].startswith("keeps control for min_hold"):
        return [f"{name} asked ({reason}) but {agent} keeps control for at least the minimum hold time, "
                f"so the lights do not flicker; only a higher-priority agent may interrupt it."
                for name, _, reason in others] or [f"{agent} keeps control for the minimum hold time."]
    if not claims:
        return ["No specialist asked for control, so the default controller decided."]
    reasons = []
    for name, urgency, reason in others:
        if rank.get(name, len(priority)) > rank.get(agent, len(priority)):
            reasons.append(f"{name} also asked ({reason}) but has a lower priority than {agent} "
                           f"(order: {' > '.join(priority)}).")
        else:
            reasons.append(f"{name} also asked ({reason}) with a lower urgency ({urgency:.2f}).")
    return reasons


def what_if(row: dict) -> str:
    """Counterfactual part: what a fixed timer would have done."""
    would = row.get("fixed_time_would", "")
    if would == "":
        return ""
    if int(would) == row["final"]:
        return "A fixed timer would have chosen the same phase."
    return f"A fixed timer would have chosen phase {would} instead of phase {row['final']}."


def explain_decision(row: dict, priority: list[str]) -> str:
    """One decision as a short paragraph: who, why, why not, what if, shield, warnings."""
    parts = [f"At t={row['time']:.0f} s, junction {row['tls_id']} gave green phase {row['final']}; "
             f"{row['agent']} was in control because: {row['reason']}."]
    if row["explanation"]:
        parts.append(f"Its choice: {row['explanation']}.")
    parts += why_not(row, priority)
    if what_if(row):
        parts.append(what_if(row))
    if row["shield_note"]:
        parts.append(f"The safety shield changed it: {row['shield_note']}.")
    else:
        parts.append("The safety shield approved it.")
    if row.get("events"):
        parts.append(f"Coordinator warnings at that moment: {row['events']}.")
    return " ".join(parts)


def cite(row: dict) -> str:
    return f"[{row['tls_id']} t={row['time']:.0f}]"


def fact_line(row: dict, priority: list[str]) -> str:
    """One decision as compact "field = value" facts (easier for a small LLM than prose)."""
    others = [c for c in parse_claims(row.get("claims", "")) if c[0] != row["agent"]]
    lost = ", ".join(f"{n} (lost: {_why_lost(n, u, row, priority)})" for n, u, _ in others) or "none"
    would = row.get("fixed_time_would", "")
    fields = [
        f"in control = {row['agent']}",
        f"because = {row['reason']}",
        f"green phase given = {row['final']}",
        f"action = {row['explanation']}",
        f"other agents that asked = {lost}",
        f"fixed timer would choose = {'same phase' if would == '' or int(would) == row['final'] else 'phase ' + would}",
        f"safety shield = {'changed: ' + row['shield_note'] if row['shield_note'] else 'approved'}",
    ]
    if row.get("events"):
        fields.append(f"warnings = {row['events']}")
    return f"{cite(row)} " + "; ".join(fields)


def _why_lost(name: str, urgency: float, row: dict, priority: list[str]) -> str:
    if row["reason"].startswith("keeps control for min_hold"):
        return "minimum hold time of the agent in control"
    rank = {n: i for i, n in enumerate(priority)}
    if rank.get(name, len(priority)) > rank.get(row["agent"], len(priority)):
        return "lower priority"
    return f"lower urgency {urgency:.2f}"


# --------------------------------------------------------------------------- retrieval
def parse_question(question: str, agents: list[str]) -> dict:
    """Pull junction, time, vehicle and agent names out of a question."""
    q = question.lower()
    junctions = [j.upper() for j in re.findall(r"\bj\d+\b", q)]
    vehicles = re.findall(r"\b(amb_\d+|accident_\d+|bus_\w+\.\d+)\b", q)
    # remove ids and "phase N" so that only times are left as numbers
    no_ids = re.sub(r"\bj\d+\b|\b(amb|accident)_\d+\b|bus_\w+\.\d+|\bphases?\s+\d+", " ", q)
    times = [float(t) for t in re.findall(r"\b(\d{1,5}(?:\.\d+)?)\s*(?:s\b|sec|seconds?)?", no_ids)]
    named = [a for a in agents if a in q] + (["spillback"] if "spillback" in q else [])
    return {"junctions": junctions, "times": times, "vehicles": vehicles, "agents": named}


def find_rows(rows: list[dict], question: str, agents: list[str], max_rows: int = 6) -> list[dict]:
    """The log rows a question is about, at most max_rows (earliest first)."""
    query = parse_question(question, agents)
    pool = [r for r in rows if not query["junctions"] or r["tls_id"] in query["junctions"]]
    if query["vehicles"]:
        # decisions ABOUT the vehicle first; only warnings that mention it if there are none
        about = [r for r in pool if any(v in (r["reason"] + r["explanation"]).lower() for v in query["vehicles"])]
        pool = about or [r for r in pool if any(v in r["events"].lower() for v in query["vehicles"])]
    for name in query["agents"]:
        if name == "spillback":
            pool = [r for r in pool if "spillback" in r["events"]]
        else:
            pool = [r for r in pool if r["agent"] == name or name in r.get("claims", "")]
    if query["times"]:
        # per junction, the decision closest in time to each asked time
        picked = []
        for t in query["times"]:
            for tls in sorted({r["tls_id"] for r in pool}):
                candidates = [r for r in pool if r["tls_id"] == tls]
                if candidates:
                    picked.append(min(candidates, key=lambda r: abs(r["time"] - t)))
        pool = picked
    elif not (query["vehicles"] or query["agents"]):
        return []  # nothing specific to look for
    unique = {(r["tls_id"], r["time"]): r for r in pool}
    return sorted(unique.values(), key=lambda r: (r["time"], r["tls_id"]))[:max_rows]


def template_answer(question: str, rows: list[dict], priority: list[str]) -> str:
    """Answer without LLM: the explanation of every retrieved decision, with citations."""
    if not rows:
        return ("I found no decision in the log matching this question. Mention a junction (J1, J2, J3), "
                "a time in seconds, a vehicle (amb_1) or an agent (emergency, flow, operator, spillback).")
    return "\n".join(f"{cite(r)} {explain_decision(r, priority)}" for r in rows)


# --------------------------------------------------------------------------- run story
def segments(rows: list[dict], tls: str) -> list[tuple[str, float, float, str]]:
    """Consecutive decisions of one junction with the same agent: (agent, start, end, first reason)."""
    result = []
    for r in (r for r in rows if r["tls_id"] == tls):
        if result and result[-1][0] == r["agent"]:
            agent, start, _, reason = result[-1]
            result[-1] = (agent, start, r["time"], reason)
        else:
            result.append((r["agent"], r["time"], r["time"], r["reason"]))
    return result


def run_story(rows: list[dict]) -> list[str]:
    """The important moments of a run, in time order, as sentences."""
    lines = []
    for tls in sorted({r["tls_id"] for r in rows}):
        for agent, start, end, reason in segments(rows, tls):
            if agent not in DEFAULT_AGENTS:
                lines.append((start, f"[{tls} t={start:.0f}-{end:.0f}] {agent} took control: {reason}."))
        spill = [r["time"] for r in rows if r["tls_id"] == tls and "spillback" in r["events"]]
        if spill:
            lines.append((spill[0], f"[{tls} t={spill[0]:.0f}-{spill[-1]:.0f}] spillback warning: the road "
                                    f"after {tls} was filling up ({len(spill)} decisions)."))
        for r in rows:
            if r["tls_id"] == tls and r["shield_note"]:
                lines.append((r["time"], f"{cite(r)} the safety shield changed {r['agent']}'s choice: "
                                         f"{r['shield_note']}."))
    story = [text for _, text in sorted(lines)]

    total = len(rows) or 1
    counts = {}
    for r in rows:
        counts[r["agent"]] = counts.get(r["agent"], 0) + 1
    shares = ", ".join(f"{a} {100 * n / total:.0f}%" for a, n in sorted(counts.items(), key=lambda x: -x[1]))
    compared = [r for r in rows if r.get("fixed_time_would", "") != ""]
    differs = sum(int(r["fixed_time_would"]) != r["final"] for r in compared)
    story.append(f"Control share over the run: {shares}.")
    if compared:
        story.append(f"Our phase differed from what a fixed timer would have chosen in "
                     f"{100 * differs / len(compared):.0f}% of the decisions.")
    return story
