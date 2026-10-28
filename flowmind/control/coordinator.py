"""Corridor coordinator: warns the junctions of the corridor about what is coming.

Owner: Adem.

Every step it rebuilds the list of warnings (Event objects, see contracts.py):
- emergency: for each ambulance and each junction still ahead of it,
  ETA = distance / speed. If ETA < coordinator.horizon, the junction gets
  Event("emergency", data={vehicle_id, eta, distance, lane_id}). lane_id is the
  lane the ambulance will arrive on, so the junction can give it green BEFORE
  the ambulance is even on its street ("green corridor").
- incident: when the incident agent controls a junction, its neighbours on the
  corridor get Event("incident") so they stop feeding the blocked area. Only one
  hop: a junction that was itself warned about an incident does not pass the
  warning on, otherwise two neighbours warn each other forever.
- spillback: when the road between two neighbouring junctions A -> B is filling
  up (occupancy above coordinator.spillback_occupancy), A gets
  Event("incident", data={"cause": "spillback", ...}) so it stops sending more
  cars into it. The link is found by lane id: A's outgoing lanes that are also
  B's incoming lanes.

Warnings are recomputed every step: when the ambulance has passed (or stopped
being expected), its warning simply disappears. That is the cancellation.
"""
from __future__ import annotations

from flowmind.contracts import Event


class Coordinator:
    def __init__(self, params: dict, order: list[str]):
        """params: coordinator section of config.yaml; order: junctions along the corridor."""
        self.params = params
        self.order = order
        self.warned: set[str] = set()   # junctions warned about an incident last step

    def update(self, time: float, approaches: list, decisions: list,
               states: dict | None = None) -> dict[str, list[Event]]:
        """approaches: env.emergency_approaches(); decisions: last step's Decisions;
        states: current JunctionStates by tls_id (needed for the spillback warning)."""
        events = {tls_id: [] for tls_id in self.order}

        # Emergency vehicles: warn every junction they will reach within the horizon.
        for vehicle_id, tls_id, distance, speed, lane_id in approaches:
            eta = distance / max(speed, self.params["min_speed"])
            if tls_id in events and eta <= self.params["horizon"]:
                events[tls_id].append(Event(
                    kind="emergency",
                    target_tls=tls_id,
                    expires_at=time + eta + self.params["margin"],
                    data={"vehicle_id": vehicle_id, "eta": round(eta, 1),
                          "distance": round(distance, 1), "lane_id": lane_id},
                ))

        # Incidents: warn the neighbours of a junction where the incident agent is in control.
        for d in decisions:
            if d.agent != "incident" or d.tls_id not in self.order or d.tls_id in self.warned:
                continue   # a warned junction only relays: no warning back (no loop)
            i = self.order.index(d.tls_id)
            for neighbour in self.order[max(i - 1, 0):i] + self.order[i + 1:i + 2]:
                events[neighbour].append(Event(
                    kind="incident",
                    target_tls=neighbour,
                    expires_at=time + self.params["incident_ttl"],
                    data={"cause": "incident", "from_tls": d.tls_id},
                ))

        # Spillback: warn the junction that feeds a link which is filling up.
        for a, b in self.neighbour_pairs():
            if not states or a not in states or b not in states:
                continue
            link = {lane.lane_id for lane in states[a].out_lanes}
            occupancy = max((lane.occupancy for lane in states[b].lanes if lane.lane_id in link), default=0.0)
            if occupancy >= self.params["spillback_occupancy"]:
                events[a].append(Event(
                    kind="incident",
                    target_tls=a,
                    expires_at=time + self.params["incident_ttl"],
                    data={"cause": "spillback", "from_tls": b, "occupancy": round(occupancy, 2),
                          "reason": f"road {a}->{b} is {occupancy:.0%} full"},
                ))
        self.warned = {t for t, evs in events.items()
                       if any(e.kind == "incident" and e.data.get("cause") == "incident" for e in evs)}
        return events

    def neighbour_pairs(self) -> list[tuple[str, str]]:
        """(A, B) for every pair of neighbouring junctions, in both directions."""
        forward = list(zip(self.order, self.order[1:]))
        return forward + [(b, a) for a, b in forward]
