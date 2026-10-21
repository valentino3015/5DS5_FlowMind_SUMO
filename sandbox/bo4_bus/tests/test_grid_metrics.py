import math
import os
import sys

import numpy as np
import pytest

BO4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BO4)
import bus_metrics as bm  # noqa: E402


def _stop(veh, stop, started, ended):
    return {"id": veh, "busStop": stop, "started": started, "ended": ended}


def _trip(veh, depart, arrival, time_loss, depart_delay=0.0, unfinished=False):
    return {"id": veh, "depart": depart, "arrival": arrival, "timeLoss": time_loss,
            "departDelay": depart_delay, "unfinished": unfinished}


def test_line_headway_profile_in_stop_order():
    # regular at s0 (180, 180), bunched at s1 (60 < 90, 300)
    stops = [_stop("b0", "s0", 1000, 1020), _stop("b1", "s0", 1180, 1200),
             _stop("b2", "s0", 1360, 1380),
             _stop("b0", "s1", 1100, 1120), _stop("b1", "s1", 1160, 1190),
             _stop("b2", "s1", 1460, 1480)]
    prof = bm.line_headway_profile(stops, ["s0", "s1"], 180, 600, 7800)
    assert [p["stop"] for p in prof] == ["s0", "s1"]
    assert prof[0]["cv"] == 0.0 and prof[0]["bunching"] == 0
    assert prof[1]["n_headways"] == 2 and prof[1]["bunching"] == 1
    assert prof[1]["cv"] > prof[0]["cv"]


def test_cv_trend_slope():
    assert bm.cv_trend([0.2, 0.3, 0.4, 0.5]) == pytest.approx(0.1)
    assert bm.cv_trend([0.5, 0.5, 0.5]) == pytest.approx(0.0)
    assert bm.cv_trend([0.2, math.nan, 0.4]) == pytest.approx(0.1)   # nan skipped, x kept
    assert math.isnan(bm.cv_trend([0.3]))


def test_segment_delays_one_per_junction():
    stops = [_stop("b", "s0", 700, 720), _stop("b", "s1", 780, 800), _stop("b", "s2", 900, 925)]
    segs = bm.segment_delays(stops, ["s0", "s1", "s2"], [30.0, 40.0], 600, 7800)
    assert segs == [{"b": 30.0}, {"b": 60.0}]


def test_travel_times_window_and_unfinished():
    trips = [_trip("b1", 700, 1000, 50), _trip("b2", 100, 400, 50),          # before window
             _trip("b3", 7700, -1, 50, unfinished=True), _trip("car", 700, 900, 5)]
    assert bm.travel_times(trips, {"b1", "b2", "b3"}, 600, 7800) == {"b1": 300}


def test_car_total_delay_includes_depart_delay():
    trips = [_trip("c1", 700, 900, 20, 10), _trip("c2", 800, -1, 60, 0, unfinished=True),
             _trip("bus", 700, 900, 100, 0)]
    assert bm.car_mean_total_delay(trips, {"bus"}, 600, 7800) == pytest.approx(45.0)


def test_person_delay_with_depart_delay_field():
    trips = [_trip("bus", 700, 900, 10, 2), _trip("car", 700, 900, 10, 2)]
    old = bm.person_delay(trips, {"bus": 40}, 1.3, 600, 7800)
    new = bm.person_delay(trips, {"bus": 40}, 1.3, 600, 7800, fields=("timeLoss", "departDelay"))
    assert old == pytest.approx(413.0)                 # default unchanged
    assert new == pytest.approx(12 * 40 + 12 * 1.3)


def test_longest_run():
    assert bm.longest_run([False, True, True, False, True, True, True], dt=5) == 15
    assert bm.longest_run([], dt=5) == 0
    assert bm.longest_run([False, False]) == 0


def test_closed_loop_dwell_formula():
    sys.path.insert(0, os.path.join(BO4, "nets", "grid"))
    import Grid_Demo as gd
    lam = (gd.MEAN_DWELL - gd.T_FIXED) / (gd.T_BOARD * 180.0)
    rng = np.random.default_rng(0)
    nominal = [gd.dwell_time(rng, 180.0, lam)[0] for _ in range(20000)]
    assert np.mean(nominal) == pytest.approx(25.0, abs=0.2)     # mean 25 s at gap = H
    d, b = gd.dwell_time(rng, 0.0, lam)
    assert (d, b) == (10.0, 0)                                  # no gap -> fixed time only
    late = [gd.dwell_time(rng, 360.0, lam)[0] for _ in range(20000)]
    assert np.mean(late) == pytest.approx(40.0, abs=0.3)        # 10 + 2.5 * 12


def test_approach_delay_pools_pieces_and_counts_last_piece(tmp_path):
    p = tmp_path / "edgedata.xml"
    p.write_text('<meandata><interval begin="600" end="7800" id="cars">'
                 '<edge id="a" timeLoss="100" left="10"/><edge id="a.250.00" timeLoss="300" left="20"/>'
                 '<edge id="b.250.00" timeLoss="50" left="5"/></interval></meandata>')
    ed = bm.parse_edgedata(str(p))
    # (100 + 300 + 0 + 50) s over 20 + 5 cars that left the approaches
    assert bm.approach_delay(ed, {"a": ["a", "a.250.00"], "b": ["b", "b.250.00"]}) == 18.0
    assert bm.approach_delay(ed, {"a": ["a", "a.250.00"]}) == 20.0
    assert math.isnan(bm.approach_delay(ed, {"c": ["c", "c.250.00"]}))
