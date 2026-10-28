"""Rule-based specialist that serves overdue pedestrian crossings."""
from __future__ import annotations

from flowmind.agents.base import Specialist
from flowmind.contracts import Claim, Crossing, Event, JunctionState


class PedestrianAgent(Specialist):
    name = "pedestrian"

    def wants_control(self, state: JunctionState, events: list[Event]) -> Claim | None:
        request = self._most_urgent_request(state)
        if request is None:
            return None

        phase, waiting, max_wait, crossing = request
        min_waiting = max(int(self.params.get("min_waiting", 4)), 1)
        hard_wait = max(float(self.params.get("hard_wait", 60)), 1.0)
        group_score = min(waiting / min_waiting, 1.0)
        wait_score = min(max_wait / hard_wait, 1.0)
        urgency = min(1.0, 0.3 + 0.25 * group_score + 0.45 * wait_score)
        if max_wait >= hard_wait:
            urgency = max(urgency, 0.9)

        return Claim(
            urgency=urgency,
            reason=(
                f"{waiting} pedestrians waited up to {max_wait:.0f} s "
                f"for phase {phase}, including {crossing.crossing_id}"
            ),
        )

    def act(self, state: JunctionState, events: list[Event], allowed: list[bool]) -> int:
        candidates = sorted(
            self._eligible_phase_requests(state),
            key=lambda request: (request[2], request[1]),
            reverse=True,
        )

        for phase, waiting, max_wait, crossing in candidates:
            if phase < len(allowed) and allowed[phase]:
                self.last_reason = (
                    f"phase {phase} serves {waiting} pedestrians "
                    f"waiting up to {max_wait:.0f} s, including "
                    f"{crossing.crossing_id}"
                )
                return phase

        self.last_reason = "overdue pedestrian crossing is not currently allowed"
        return self.keep_phase(state, allowed)

    def _eligible_phase_requests(
        self, state: JunctionState
    ) -> list[tuple[int, int, float, Crossing]]:
        """Aggregate compatible crossings and return eligible phase requests."""
        min_waiting = int(self.params.get("min_waiting", 4))
        large_group_priority = int(self.params.get("large_group_priority", 10))
        batch_service_after = float(self.params.get("batch_service_after", 0))
        request_wait = float(self.params.get("max_wait", 5))
        hard_wait = float(self.params.get("hard_wait", 60))
        crossing_by_id = {
            crossing.crossing_id: crossing for crossing in state.crossings
        }
        requests = []
        for phase, crossing_ids in enumerate(state.phase_crossings):
            crossings = [
                crossing_by_id[crossing_id]
                for crossing_id in crossing_ids
                if crossing_id in crossing_by_id
                and crossing_by_id[crossing_id].waiting > 0
            ]
            if not crossings:
                continue
            waiting = sum(crossing.waiting for crossing in crossings)
            most_urgent = max(
                crossings, key=lambda crossing: (crossing.max_wait, crossing.waiting)
            )
            max_wait = most_urgent.max_wait
            normal_batch_ready = (
                waiting >= min_waiting
                and max_wait >= request_wait
                and state.time_in_phase >= batch_service_after
            )
            large_batch_ready = (
                waiting >= large_group_priority and max_wait >= request_wait
            )
            if max_wait >= hard_wait or normal_batch_ready or large_batch_ready:
                requests.append((phase, waiting, max_wait, most_urgent))
        return requests

    def _most_urgent_request(
        self, state: JunctionState
    ) -> tuple[int, int, float, Crossing] | None:
        """Return the eligible phase with the greatest wait pressure."""
        candidates = self._eligible_phase_requests(state)
        if not candidates:
            return None
        return max(candidates, key=lambda request: (request[2], request[1]))
