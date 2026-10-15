"""Safety shield (the safety guard): checks every action before it reaches SUMO.

Owner: pedestrian agent owner (TBD).
Basic version written by Adem so the orchestrator can run: only the green-time
rules. The pedestrian owner extends it at the TODOs below.

Two functions:
- allowed(state, config) -> list[bool]: which green phases are safe right now.
  This list is given to the agent's act().
- check(state, action, allowed) -> (action, note): if the agent still chose a
  blocked phase, replace it and explain why (the note is logged).
"""
from __future__ import annotations

from flowmind.contracts import JunctionState


def allowed(state: JunctionState, config: dict) -> list[bool]:
    signals = config["signals"]
    only_current = [phase == state.phase for phase in range(state.num_phases)]

    # 1. While yellow / all-red is shown, or before min_green: no switch.
    if state.in_transition or state.time_in_phase < signals["min_green"]:
        return only_current

    result = [True] * state.num_phases

    # 2. After max_green: the current phase must end (if there is another one).
    if state.time_in_phase >= signals["max_green"] and state.num_phases > 1:
        result[state.phase] = False
        # TODO(pedestrian owner): allow emergency green to exceed max_green.

    # TODO(pedestrian owner): do not end a phase while pedestrians are still on a
    #   crossing it serves (crossing.on_crossing > 0, see shield.ped_clearance_speed).
    # TODO(pedestrian owner): if pedestrians at a crossing waited more than
    #   shield.ped_max_wait, only allow the phases in state.phase_crossings that serve it.

    if not any(result):  # never block everything
        return only_current
    return result


def check(state: JunctionState, action: int, allowed_phases: list[bool]) -> tuple[int, str]:
    """Return (safe action, note). The note is "" when the action was fine."""
    if 0 <= action < len(allowed_phases) and allowed_phases[action]:
        return action, ""
    safe = state.phase if allowed_phases[state.phase] else allowed_phases.index(True)
    return safe, f"phase {action} not allowed now (green time rules), used phase {safe}"
