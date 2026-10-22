"""Which inputs mattered for one decision? (perturbation-based sensitivity analysis)

Owner: Adem.

Takes the junction state and the coordinator's warnings saved at that moment
(run.py --save-states), changes ONE input at a time, asks fresh agents again and
reports the changes that flip the decision, e.g. "without amb_1, flow would
control J2 and choose phase 2".

Limits (say them in the presentation): it replays one decision, not the whole run,
and the agents start fresh (no memory, e.g. the bus agent's last passages or the
min_hold timer). If the replay does not reproduce the logged decision, the result
says so instead of pretending.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from flowmind.contracts import Crossing, Event, JunctionState, LaneState, SpecialVehicle
from flowmind.control import shield


def state_from_dict(d: dict) -> JunctionState:
    """Rebuild a JunctionState from its JSON form."""
    d = dict(d)
    d["lanes"] = tuple(LaneState(**x) for x in d["lanes"])
    d["out_lanes"] = tuple(LaneState(**x) for x in d["out_lanes"])
    d["special_vehicles"] = tuple(SpecialVehicle(**x) for x in d["special_vehicles"])
    d["crossings"] = tuple(Crossing(**x) for x in d["crossings"])
    for key in ("phase_lanes", "phase_crossings", "phase_out_lanes"):
        if key in d:
            d[key] = tuple(tuple(x) for x in d[key])
    return JunctionState(**d)


def load_state(run_dir: Path, tls_id: str, time: float) -> tuple[JunctionState, list[Event]] | None:
    """Saved state and warnings of a junction closest to `time` (None without states.jsonl)."""
    path = Path(run_dir) / "states.jsonl"
    if not path.is_file():
        return None
    best, best_gap = None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            state = d.get("state", d)  # older files stored only the state
            if state["tls_id"] == tls_id and (best_gap is None or abs(state["time"] - time) < best_gap):
                best, best_gap = d, abs(state["time"] - time)
    if best is None:
        return None
    events = [Event(**e) for e in best.get("events", [])]
    return state_from_dict(best.get("state", best)), events


def perturbations(state: JunctionState, events: list[Event], extra_cars: int):
    """One-change variants: (description, state, events)."""
    variants = []
    for i, e in enumerate(events):
        what = e.data.get("vehicle_id") or e.data.get("cause", e.kind)
        variants.append((f"without the coordinator's warning about {what}", state, events[:i] + events[i + 1:]))
    for v in state.special_vehicles:
        others = tuple(x for x in state.special_vehicles if x.vehicle_id != v.vehicle_id)
        variants.append((f"without {v.vehicle_id} on the approach", replace(state, special_vehicles=others), events))
    for i, lane in enumerate(state.lanes):
        if lane.queue > 0:
            empty = replace(lane, queue=0, vehicles=0, waiting_time=0.0, occupancy=0.0)
            variants.append((f"if lane {lane.lane_id} were empty",
                             replace(state, lanes=_swap(state.lanes, i, empty)), events))
        busier = replace(lane, queue=lane.queue + extra_cars, vehicles=lane.vehicles + extra_cars,
                         occupancy=min(1.0, lane.occupancy + 0.04 * extra_cars))
        variants.append((f"if {extra_cars} more cars waited on {lane.lane_id}",
                         replace(state, lanes=_swap(state.lanes, i, busier)), events))
    for i, c in enumerate(state.crossings):
        if c.waiting > 0:
            variants.append((f"if nobody waited at crossing {c.crossing_id}",
                             replace(state, crossings=_swap(state.crossings, i, replace(c, waiting=0, max_wait=0.0))),
                             events))
    return variants


def _swap(items: tuple, i: int, new) -> tuple:
    return items[:i] + (new,) + items[i + 1:]


def analyze(state: JunctionState, events: list[Event], make_orchestrator, config: dict,
            extra_cars: int = 5) -> dict:
    """Decision on the real inputs, then every one-change variant that changes who decides or the phase.

    make_orchestrator(config, movements) builds fresh agents (scripts/run.py has it)."""
    def decide(s: JunctionState, ev: list[Event]):
        orchestrator = make_orchestrator(config, {})
        orchestrator.since = -1e9  # fresh: no min_hold history
        return orchestrator.decide(s, ev, shield.allowed(s, config))

    base = decide(state, events)
    variants = perturbations(state, events, extra_cars)
    changes = []
    for description, s, ev in variants:
        d = decide(s, ev)
        if (d.agent, d.final) != (base.agent, base.final):
            changes.append(f"{description}: {d.agent} would control {state.tls_id} and choose phase {d.final} "
                           f"(instead of {base.agent}, phase {base.final}).")
    return {"agent": base.agent, "phase": base.final, "reason": base.reason, "changes": changes,
            "tested": len(variants)}


def sensitivity_text(result: dict, tls_id: str, time: float, logged: dict | None = None) -> str:
    head = (f"[{tls_id} t={time:.0f}] Replayed decision: {result['agent']} controls with phase {result['phase']} "
            f"({result['reason']}). {result['tested']} one-change variants tested.")
    if logged and (logged["agent"], int(logged["final"])) != (result["agent"], result["phase"]):
        head += (f"\nNote: in the real run {logged['agent']} decided phase {logged['final']}; the replay differs "
                 f"because the agents start without memory (e.g. min_hold or a bus's history), so read the "
                 f"result below with care.")
    if not result["changes"]:
        return head + "\nNone of the single changes alters the decision: it is robust to each input."
    return head + "\nInputs that matter:\n- " + "\n- ".join(result["changes"])
