"""Key Performance Indicators (KPIs) for FlowMind traffic simulations.

Tracks, computes, and reports the 5 core traffic engineering metrics:
1. Total vehicles that arrived at their destinations (throughput).
2. Average and total waiting time (queue delay).
3. Average travel time (trip duration).
4. Average speed over the entire run.
5. Cumulative halting or delay over time.

Owner: Ahmed / Evaluation team.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


@dataclass
class SimulationKPIs:
    """Consolidated KPI report for a simulation run."""

    scenario: str
    controller: str
    duration_seconds: float
    total_steps: int

    # 1. Throughput
    total_departed: int          # Total vehicles that entered the network
    total_arrived: int           # Total vehicles that reached their destination
    arrival_rate: float          # Percentage of vehicles that completed trips (%)
    active_vehicles: int         # Vehicles still in the network at end

    # 2. Waiting Time
    total_waiting_time: float    # Cumulative waiting time across all vehicles (s)
    average_waiting_time: float  # Mean waiting time per vehicle (s)

    # 3. Travel Time
    average_travel_time: float   # Mean trip duration for completed trips (s)

    # 4. Speed
    average_speed: float         # Mean speed over the entire run (m/s)
    average_speed_kmh: float     # Mean speed in km/h

    # 5. Halting & Delay
    cumulative_halting: float    # Cumulative vehicle-seconds of halting (veh*s)
    total_box_vehicles: int      # Cumulative occurrences of vehicles in intersection box

    # 6. Travel-time predictability (default=0.0 so old code that constructs this stays valid)
    p50_travel_time: float = 0.0  # Median trip duration (s)
    p95_travel_time: float = 0.0  # 95th-percentile trip duration (s) — worst-case travellers
    travel_time_std: float = 0.0  # Standard deviation of trip durations (predictability)

    def to_dict(self) -> dict:
        return asdict(self)



class KPITracker:
    """Real-time and post-simulation KPI accumulator."""

    def __init__(self, scenario: str = "3x3grid", controller: str = "flow"):
        self.scenario = scenario
        self.controller = controller

        self.depart_times: dict[str, float] = {}
        self.travel_times: list[float] = []

        self.total_departed: int = 0
        self.total_arrived: int = 0

        self.step_speeds: list[float] = []
        self.step_haltings: list[int] = []
        self.step_box_vehicles: list[int] = []

        self.cumulative_halting_seconds: float = 0.0
        self.total_waiting_seconds: float = 0.0

        self.duration_seconds: float = 0.0
        self.total_steps: int = 0

    def reset(self):
        """Reset all accumulators for a new simulation."""
        self.depart_times.clear()
        self.travel_times.clear()
        self.total_departed = 0
        self.total_arrived = 0
        self.step_speeds.clear()
        self.step_haltings.clear()
        self.step_box_vehicles.clear()
        self.cumulative_halting_seconds = 0.0
        self.total_waiting_seconds = 0.0
        self.duration_seconds = 0.0
        self.total_steps = 0

    def update_step(
        self,
        sim_time: float,
        delta_time: float,
        departed_ids: list[str],
        arrived_ids: list[str],
        halting_count: int,
        mean_speed: float,
        box_count: int = 0,
    ):
        """Record metrics from one simulation decision step."""
        self.total_steps += 1
        self.duration_seconds += delta_time

        # Track vehicle departures and arrivals
        for v in departed_ids:
            self.depart_times[v] = sim_time
            self.total_departed += 1

        for v in arrived_ids:
            self.total_arrived += 1
            if v in self.depart_times:
                self.travel_times.append(float(sim_time - self.depart_times.pop(v)))

        # Record halting & speed metrics
        self.step_haltings.append(halting_count)
        self.step_speeds.append(mean_speed)
        self.step_box_vehicles.append(box_count)

        # Cumulative halting in vehicle-seconds
        step_halting_seconds = float(halting_count * delta_time)
        self.cumulative_halting_seconds += step_halting_seconds
        self.total_waiting_seconds += step_halting_seconds

    def compute(self, tripinfo_file: str | Path | None = None) -> SimulationKPIs:
        """Compute final KPIs, optionally enriching with ground-truth tripinfo XML from SUMO."""
        avg_travel = float(np.mean(self.travel_times)) if self.travel_times else 0.0
        tot_wait = self.total_waiting_seconds
        avg_wait = (tot_wait / max(1, self.total_departed)) if self.total_departed else 0.0

        # Travel-time percentiles (predictability metrics)
        if self.travel_times:
            arr = np.array(self.travel_times)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            tt_std = float(np.std(arr))
        else:
            p50 = p95 = tt_std = 0.0

        # If SUMO generated a tripinfo XML, parse exact ground-truth values
        if tripinfo_file and Path(tripinfo_file).is_file():
            try:
                tree = ET.parse(str(tripinfo_file))
                trips = tree.getroot().findall("tripinfo")
                if trips:
                    xml_durations = [float(t.attrib.get("duration", 0)) for t in trips]
                    xml_waits = [float(t.attrib.get("waitingTime", 0)) for t in trips]
                    avg_travel = float(np.mean(xml_durations))
                    tot_wait = float(np.sum(xml_waits))
                    avg_wait = float(np.mean(xml_waits))
                    # Recompute percentiles from ground-truth XML data
                    arr_xml = np.array(xml_durations)
                    p50 = float(np.percentile(arr_xml, 50))
                    p95 = float(np.percentile(arr_xml, 95))
                    tt_std = float(np.std(arr_xml))
            except Exception:
                pass

        overall_speed = float(np.mean(self.step_speeds)) if self.step_speeds else 0.0
        arrival_rate = (self.total_arrived / max(1, self.total_departed) * 100.0) if self.total_departed else 0.0

        return SimulationKPIs(
            scenario=self.scenario,
            controller=self.controller,
            duration_seconds=self.duration_seconds,
            total_steps=self.total_steps,
            total_departed=self.total_departed,
            total_arrived=self.total_arrived,
            arrival_rate=arrival_rate,
            active_vehicles=len(self.depart_times),
            total_waiting_time=tot_wait,
            average_waiting_time=avg_wait,
            average_travel_time=avg_travel,
            p50_travel_time=p50,
            p95_travel_time=p95,
            travel_time_std=tt_std,
            average_speed=overall_speed,
            average_speed_kmh=overall_speed * 3.6,
            cumulative_halting=self.cumulative_halting_seconds,
            total_box_vehicles=int(np.sum(self.step_box_vehicles)),
        )


def print_kpi_report(kpis: SimulationKPIs, title: str = "FlowMind Traffic KPI Report"):
    """Print an executive, formatted KPI table in the terminal."""
    bar = "=" * 64
    sub = "-" * 64
    print("\n" + bar)
    print(f"            {title.upper()}            ")
    print(bar)
    print(f"Scenario            : {kpis.scenario}")
    print(f"Controller          : {kpis.controller}")
    print(f"Simulation Duration : {kpis.duration_seconds:.0f}s ({kpis.total_steps} decision steps)")
    print(sub)
    print(f"1. Total Arrived    : {kpis.total_arrived} vehicles "
          f"({kpis.arrival_rate:.1f}% of {kpis.total_departed} departed)")
    print(f"   Active in Grid   : {kpis.active_vehicles} vehicles")
    print(sub)
    print(f"2. Waiting Time     :")
    print(f"   - Total Waiting  : {kpis.total_waiting_time:,.1f} s")
    print(f"   - Average Waiting: {kpis.average_waiting_time:.2f} s / vehicle")
    print(sub)
    print(f"3. Avg Travel Time  : {kpis.average_travel_time:.2f} s (mean, completed trips)")
    print(f"   Median (P50)     : {kpis.p50_travel_time:.2f} s")
    print(f"   95th-Pct (P95)   : {kpis.p95_travel_time:.2f} s  (worst-case travellers)")
    print(f"   Std Dev          : {kpis.travel_time_std:.2f} s  (predictability)")
    print(sub)
    print(f"4. Average Speed    : {kpis.average_speed:.2f} m/s ({kpis.average_speed_kmh:.1f} km/h)")
    print(sub)
    print(f"5. Cumulative Delay :")
    print(f"   - Total Halting  : {kpis.cumulative_halting:,.1f} veh*s")
    print(f"   - Box Incidents  : {kpis.total_box_vehicles} vehicle occurrences")
    print(bar + "\n")
