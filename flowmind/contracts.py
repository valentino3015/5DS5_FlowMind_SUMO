"""Shared data types: what every agent receives and returns.

Owner: Adem. Shared by the whole team: do not change without telling everyone.

Every 5 seconds, env.py reads SUMO and builds one JunctionState per traffic light.
Agents read it (they never talk to SUMO themselves) and answer with a Claim
("I want control") and a green phase index ("give green to phase 2").

Units everywhere: seconds (s), metres (m), metres per second (m/s).
All classes are frozen: once created, a state cannot be changed by accident.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LaneState:
    """Traffic on one lane, measured right now."""

    lane_id: str          # SUMO lane id, e.g. "north_in_0"
    queue: int            # number of halting vehicles (speed < 0.1 m/s)
    vehicles: int         # number of vehicles on the lane (moving or not)
    waiting_time: float   # total waiting time of the vehicles on the lane (s)
    occupancy: float      # share of the lane covered by vehicles, 0..1


@dataclass(frozen=True)
class SpecialVehicle:
    """A bus or an emergency vehicle approaching the junction."""

    vehicle_id: str       # SUMO vehicle id, e.g. "bus_12"
    kind: str             # "bus" or "emergency"
    lane_id: str          # lane it is on now
    distance: float       # distance to the stop line (m)
    speed: float          # current speed (m/s)
    delay: float | None   # buses: seconds late (negative = early); None if unknown or not a bus


@dataclass(frozen=True)
class Crossing:
    """A pedestrian crossing at the junction."""

    crossing_id: str      # SUMO crossing id, e.g. ":J1_c0"
    waiting: int          # pedestrians waiting to cross
    max_wait: float       # longest waiting time among them (s)
    on_crossing: int      # pedestrians currently walking on the crossing


@dataclass(frozen=True)
class JunctionState:
    """Everything an agent knows about its junction at one decision step."""

    tls_id: str                                   # traffic light id, e.g. "J1"
    time: float                                   # simulation time (s)
    num_phases: int                               # number of GREEN phases (indices 0..num_phases-1)
    phase: int                                    # current green phase index
    time_in_phase: float                          # time since this green phase started (s)
    in_transition: bool                           # True while yellow / all-red is shown
    lanes: tuple[LaneState, ...]                  # incoming lanes
    out_lanes: tuple[LaneState, ...]              # outgoing lanes
    special_vehicles: tuple[SpecialVehicle, ...]  # buses and emergency vehicles approaching
    crossings: tuple[Crossing, ...]               # pedestrian crossings


@dataclass(frozen=True)
class Claim:
    """An agent's answer to "do you need control now?"."""

    urgency: float        # 0 = not urgent, 1 = most urgent
    reason: str           # one short sentence, e.g. "bus_12 is 70 s late, 120 m away"


@dataclass(frozen=True)
class Event:
    """A warning posted by the corridor coordinator for another junction."""

    kind: str             # "emergency" or "incident"
    target_tls: str       # junction that should react, e.g. "J2"
    expires_at: float     # simulation time after which the event is ignored (s)
    data: dict = field(default_factory=dict)  # details, e.g. {"vehicle_id": "amb_1", "eta": 18.0}


# ---------------------------------------------------------------------------
# Example data, so you can test your agent before the SUMO connection exists.
# ---------------------------------------------------------------------------

def make_example_state() -> JunctionState:
    """Return a realistic hand-made JunctionState.

    The junction "J1" has 4 approaches (north, south, east, west), each with a
    straight lane (_0) and a left-turn lane (_1), and 4 green phases:
        0 = north-south straight   1 = north-south left
        2 = east-west straight     3 = east-west left
    Right now phase 2 (east-west straight) has been green for 14 s.
    North is congested, a bus is late, pedestrians have waited long.
    """
    def lane(lane_id, queue, vehicles, waiting_time, occupancy):
        return LaneState(lane_id, queue, vehicles, waiting_time, occupancy)

    lanes = (
        lane("north_in_0", 8, 10, 95.0, 0.55),
        lane("north_in_1", 3, 4, 40.0, 0.20),
        lane("south_in_0", 5, 7, 60.0, 0.40),
        lane("south_in_1", 1, 1, 12.0, 0.05),
        lane("east_in_0", 2, 6, 8.0, 0.30),
        lane("east_in_1", 0, 1, 0.0, 0.05),
        lane("west_in_0", 1, 5, 4.0, 0.25),
        lane("west_in_1", 0, 0, 0.0, 0.00),
    )
    out_lanes = (
        lane("north_out", 0, 3, 0.0, 0.15),
        lane("south_out", 0, 2, 0.0, 0.10),
        lane("east_out", 1, 4, 3.0, 0.20),
        lane("west_out", 0, 2, 0.0, 0.10),
    )
    special_vehicles = (
        SpecialVehicle("bus_12", "bus", "north_in_0", distance=120.0, speed=8.3, delay=70.0),
    )
    crossings = (
        Crossing(":J1_c0", waiting=4, max_wait=80.0, on_crossing=0),
        Crossing(":J1_c1", waiting=0, max_wait=0.0, on_crossing=1),
    )
    return JunctionState(
        tls_id="J1",
        time=1200.0,
        num_phases=4,
        phase=2,
        time_in_phase=14.0,
        in_transition=False,
        lanes=lanes,
        out_lanes=out_lanes,
        special_vehicles=special_vehicles,
        crossings=crossings,
    )


def example_phase_movements() -> dict[int, list[tuple[str, str]]]:
    """Which (incoming lane -> outgoing lane) moves get green in each phase of the example.

    Max-pressure needs this. For the real junctions, env.py will read it from SUMO.
    """
    return {
        0: [("north_in_0", "south_out"), ("south_in_0", "north_out")],
        1: [("north_in_1", "east_out"), ("south_in_1", "west_out")],
        2: [("east_in_0", "west_out"), ("west_in_0", "east_out")],
        3: [("east_in_1", "south_out"), ("west_in_1", "north_out")],
    }
