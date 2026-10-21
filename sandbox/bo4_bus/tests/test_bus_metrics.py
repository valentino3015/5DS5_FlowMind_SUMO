import math
import os
import statistics
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bus_metrics as bm  # noqa: E402


def test_headways_sorts_times():
    assert bm.headways([480, 0, 240, 700]) == [240, 240, 220]


def test_cv_regular_service_is_zero():
    assert bm.cv([240, 240, 240, 240]) == 0.0


def test_cv_hand_computed():
    hws = [100, 300, 200]  # mean 200, sample std 100
    assert bm.cv(hws) == pytest.approx(0.5)
    assert bm.cv(hws) == pytest.approx(statistics.stdev(hws) / statistics.mean(hws))


def test_cv_needs_two_values():
    assert math.isnan(bm.cv([240]))
    assert math.isnan(bm.cv([]))


def test_bunching_threshold_is_strict():
    H = 240
    hws = [240, 119, 120, 121, 30, 400]  # threshold 120: only 119 and 30 count
    assert bm.count_bunching(hws, H) == 2


def test_bunching_custom_factor():
    assert bm.count_bunching([100, 150, 200], 240, factor=0.75) == 2


def test_headway_stats_window_and_stop():
    stops = [
        {"id": "b0", "busStop": "EB_down", "started": 500.0, "ended": 520.0},   # before window
        {"id": "b1", "busStop": "EB_down", "started": 700.0, "ended": 720.0},
        {"id": "b2", "busStop": "EB_down", "started": 760.0, "ended": 790.0},   # bunched (60 s)
        {"id": "b3", "busStop": "EB_down", "started": 1060.0, "ended": 1080.0},
        {"id": "b4", "busStop": "EB_up", "started": 1100.0, "ended": 1120.0},   # other stop
    ]
    hs = bm.headway_stats(stops, "EB_down", 240, t0=600, t1=4200)
    assert hs["headways"] == [60.0, 300.0]
    assert hs["bunching"] == 1
    assert hs["cv"] == pytest.approx(statistics.stdev([60, 300]) / 180)


def test_bus_junction_delay():
    stops = [
        {"id": "b1", "busStop": "EB_up", "started": 680.0, "ended": 700.0},
        {"id": "b1", "busStop": "EB_down", "started": 760.0, "ended": 780.0},
        {"id": "b2", "busStop": "EB_up", "started": 500.0, "ended": 590.0},  # left before t0
        {"id": "b2", "busStop": "EB_down", "started": 640.0, "ended": 660.0},
    ]
    assert bm.bus_junction_delays(stops, "EB_up", "EB_down", 25.0, 600, 4200) == {"b1": 35.0}


def test_person_delay_uses_bus_occupancy():
    trips = [
        {"id": "bus", "depart": 700, "timeLoss": 10.0},
        {"id": "car", "depart": 700, "timeLoss": 10.0},
        {"id": "old", "depart": 100, "timeLoss": 99.0},  # before window
    ]
    assert bm.person_delay(trips, {"bus": 40}, 1.3, 600, 4200) == pytest.approx(413.0)
