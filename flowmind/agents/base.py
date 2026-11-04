"""The Specialist base class: the 3 functions every agent must write.

Owner: Adem (shared by all agent owners; do not change without telling everyone).

How your agent is used, every 5 seconds, for each junction:
    1. The orchestrator asks every agent: wants_control(state, events)
    2. It picks ONE agent (priority: emergency, incident, pedestrian, bus, flow).
    3. It asks that agent: act(state, events, allowed) -> green phase index
    4. The shield checks the answer; the logger saves explain().

To write your agent: copy the style of baselines.py, inherit from Specialist,
and fill in the 3 functions. Read your numbers from self.params (they come
from your section in config.yaml), never write them in the code.
"""
from __future__ import annotations

from flowmind.contracts import Claim, Event, JunctionState


class Specialist:
    """One expert that can drive a traffic light in one kind of situation."""

    # Short name used in logs and in config.yaml, e.g. "bus". Override it.
    name = "base"

    def __init__(self, params: dict | None = None):
        """params: your section of config.yaml, e.g. {"min_delay": 60}."""
        self.params = params or {}
        self.last_reason = ""  # set it in act(), returned by explain()

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        """Do I need control of this junction right now?

        state:  what is happening at the junction (see contracts.py)
        events: warnings from the coordinator for this junction (may be empty)

        Return None if you do not need control.
        Return Claim(urgency, reason) if you do: urgency between 0 and 1,
        reason is one short sentence, e.g. "bus_12 is 70 s late".
        """
        raise NotImplementedError

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        """Which green phase should be on now?

        allowed[i] is True if the shield allows green phase i right now.
        Return a green phase index i (0..state.num_phases-1) with allowed[i] True.
        Return state.phase to keep the current green.
        Only green phases: yellow and all-red are added automatically by env.py.
        Also set self.last_reason so explain() can say why.
        """
        raise NotImplementedError

    def explain(self) -> str:
        """Why did I choose my last action? One short sentence for the log."""
        return self.last_reason

    # --- small helpers you can reuse ------------------------------------
    def keep_phase(self, state: JunctionState, allowed: list[bool]) -> int:
        """Keep the current green if allowed, otherwise the first allowed phase."""
        if allowed[state.phase]:
            return state.phase
        return allowed.index(True)

    def phase_for_lane(self, state: JunctionState, lane_id: str, allowed: list[bool]) -> int | None:
        """An allowed green phase that gives green to this incoming lane, or None.

        Prefers the current phase (no switch needed), otherwise the first one found.
        Example: phase_for_lane(state, "north_in_0", allowed) -> 0
        """
        return self._phase_with(state, state.phase_lanes, lane_id, allowed)

    def phase_for_crossing(self, state: JunctionState, crossing_id: str, allowed: list[bool]) -> int | None:
        """An allowed green phase where pedestrians may walk on this crossing, or None.

        Prefers the current phase (no switch needed), otherwise the first one found.
        Example: phase_for_crossing(state, ":J1_c0", allowed) -> 0
        """
        return self._phase_with(state, state.phase_crossings, crossing_id, allowed)

    @staticmethod
    def exits_backed_up(state: JunctionState, lane_id: str, movements: dict,
                        occupancy: float, halted_share: float = 0.7) -> bool:
        """True if an exit lane this incoming lane feeds is backed up: its cars cannot move
        because the road after the junction is full (spillback), not because of this lane.

        One exit is enough: cars queue in one line, so the first car waiting for the full road
        blocks the ones behind it that would turn (W_J1_1 feeds J1_J2_1 and J1_S1_1).
        movements: {phase: [(incoming lane, outgoing lane), ...]} from env.get_phase_movements.
        Backed up = occupancy >= `occupancy` and at least `halted_share` of its cars halting.
        Measured on the corridor (3 seeds, 30 min): normal exit lanes stay below 0.32
        (99 % below 0.16); the road blocked by the accident reads 0.35-0.44.
        Without movements (or if the lane feeds no known exit) the answer is False.
        """
        exits = {dst for moves in movements.values() for src, dst in moves if src == lane_id}
        out = {lane.lane_id: lane for lane in state.out_lanes}
        lanes = [out[d] for d in exits if d in out]
        return any(
            l.occupancy >= occupancy and l.vehicles > 0 and l.queue >= halted_share * l.vehicles
            for l in lanes)

    @staticmethod
    def _phase_with(state: JunctionState, per_phase: tuple, item_id: str, allowed: list[bool]) -> int | None:
        candidates = [p for p, ids in enumerate(per_phase) if item_id in ids and allowed[p]]
        if not candidates:
            return None
        return state.phase if state.phase in candidates else candidates[0]
