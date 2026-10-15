"""
Minimal copies of the team contract dataclasses (FlowMind Team Handbook, section 6.1,
contracts v0.1 draft). Standalone, so bo4_bus/rl/ runs without the flowmind package;
to be replaced by `from flowmind.contracts import ...` when ported to
flowmind/specialists/bus.

All units SI: seconds, metres, metres per second.

PROPOSED CONTRACT CHANGE (not yet agreed by the team, see handbook rule 3):
VehicleInfo gets two optional fields for buses,
    line_id: str | None            line direction, e.g. "L1_EB"
    headway_lateness: float | None s, t - (last passage of the same line at this junction) - H;
                                   late (+) / early (-); None if no previous passage on record
The existing schedule_delay is relative to a timetable; BO4 controls headways, not a
timetable, so it is left None here.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LaneState:
    lane_id: str
    length: float                 # m
    vehicle_count: int
    halting_count: int            # vehicles below 0.1 m/s: the queue
    occupancy: float              # 0..1
    mean_speed: float             # m/s
    waiting_time: float           # s, accumulated waiting time on the lane
    co2_mg: float                 # CO2 emitted on the lane in the last second, mg


@dataclass(frozen=True)
class VehicleInfo:
    vehicle_id: str
    vclass: str                   # "passenger", "bus", "emergency", "truck", ...
    lane_id: str
    distance_to_stopline: float   # m
    speed: float                  # m/s
    waiting_time: float           # s
    schedule_delay: float | None = None    # s late (+) or early (-); buses only
    persons_on_board: int | None = None    # buses only
    # --- proposed contract change (BO4) ---
    line_id: str | None = None
    headway_lateness: float | None = None  # s late (+) / early (-) vs headway H


@dataclass(frozen=True)
class JunctionState:
    tls_id: str
    sim_time: float               # s
    green_phases: tuple           # green signal strings; action k = green_phases[k]
    current_green: int            # index of the current or last green phase
    time_in_phase: float          # s since the current green started
    in_transition: bool           # True during yellow or all-red
    incoming: tuple               # tuple[LaneState, ...]
    outgoing: tuple               # tuple[LaneState, ...]
    approaching: tuple            # tuple[VehicleInfo, ...]: special vehicles within detection range
    crossings: tuple = field(default_factory=tuple)   # CrossingState, not used in BO4
    neighbours: tuple = field(default_factory=tuple)  # adjacent controlled junctions

    def total_queue(self) -> int:
        return sum(l.halting_count for l in self.incoming)

    def total_waiting(self) -> float:
        return sum(l.waiting_time for l in self.incoming)
