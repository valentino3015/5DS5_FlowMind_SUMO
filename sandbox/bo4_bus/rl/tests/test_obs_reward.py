import os
import sys

import numpy as np
import pytest

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
import observation as ob  # noqa: E402
import reward as rw  # noqa: E402
from contract import JunctionState, LaneState, VehicleInfo  # noqa: E402
from sense import APPROACHES  # noqa: E402


def lane(lid, length=34.0, halting=0, waiting=0.0):
    return LaneState(lane_id=lid, length=length, vehicle_count=halting, halting_count=halting,
                     occupancy=0.0, mean_speed=0.0, waiting_time=waiting, co2_mg=0.0)


def bus(vid, lid, dist, late=None, occ=30, waiting=0.0, speed=0.0, at_stop=False):
    return VehicleInfo(vehicle_id=vid, vclass="bus", lane_id=lid, distance_to_stopline=dist,
                       speed=speed, waiting_time=waiting, persons_on_board=occ,
                       line_id=vid.rsplit("_", 1)[0], headway_lateness=late, at_stop=at_stop)


def state(incoming=(), approaching=(), green=0, tip=0.0, transition=False):
    return JunctionState(tls_id="B1", sim_time=0.0, green_phases=("GGrr", "rrGG"),
                         current_green=green, time_in_phase=tip, in_transition=transition,
                         incoming=tuple(incoming), outgoing=(), approaching=tuple(approaching))


def b1_lanes(halting=0, waiting=0.0):
    """2 x 234 m + 3 x 34 m lanes per approach = 76 car capacity."""
    out = []
    for a in APPROACHES:
        out += [lane(f"{a}_{i}", 234.0, halting, waiting) for i in range(2)]
        out += [lane(f"{a}.250.00_{i}", 34.0, halting, waiting) for i in range(3)]
    return out


# ------------------------------------------------------------------ observation
def test_observation_shape_dtype_and_empty_state():
    o = ob.observe(state(b1_lanes()))
    assert o.shape == (ob.OBS_SIZE,) == (24,)
    assert o.dtype == np.float32
    assert np.all(o[:20] == 0)
    assert list(o[20:]) == [1.0, 0.0, 0.0, 0.0]          # E-W green, no transition, 0 s


def test_observation_queue_phase_and_time():
    lanes = b1_lanes()
    lanes[0] = lane("A1B1_0", 234.0, halting=19)          # 19 / 76 on approach W
    o = ob.observe(state(lanes, green=1, tip=30.0, transition=True))
    assert o[0] == pytest.approx(19 / 76)
    assert o[5] == 0.0
    assert list(o[20:]) == [0.0, 1.0, 1.0, 0.5]


def test_observation_nearest_bus_per_approach():
    buses = [bus("L1_EB_03", "A1B1_0", 120.0, late=-40.0, occ=45),
             bus("L1_EB_04", "A1B1.250.00_0", 30.0, late=90.0, occ=30),   # nearest on W
             bus("L2_NB_01", "B0B1_1", 200.0, late=10.0),                 # beyond 150 m
             bus("L1_WB_02", "C1B1_0", 75.0, late=None, occ=60)]          # lateness unknown
    o = ob.observe(state(b1_lanes(), buses))
    w, e, s = o[0:5], o[5:10], o[10:15]
    assert list(w[1:]) == pytest.approx([1.0, 30 / 150, 0.5, 0.5])
    assert list(e[1:]) == pytest.approx([1.0, 0.5, 0.0, 1.0])
    assert list(s) == [0.0] * 5


def test_observation_always_in_bounds():
    lanes = b1_lanes(halting=500)                          # over capacity -> clipped
    buses = [bus("L1_EB_01", "A1B1_0", 0.0, late=10000.0, occ=500),
             bus("L1_WB_01", "C1B1_0", 10.0, late=-10000.0)]
    o = ob.observe(state(lanes, buses, tip=500.0))
    assert np.all(o >= ob.LOW) and np.all(o <= ob.HIGH)
    assert o[3] == 1.0 and o[8] == -1.0 and o[0] == 1.0 and o[-1] == 1.0


# ------------------------------------------------------------------ reward
def test_reward_vehicle_is_halting_vehicles_times_tick():
    s = state([lane("A1B1_0", halting=7), lane("C1B1_0", halting=3)])
    r, terms = rw.reward_vehicle(None, s)
    assert r == -50.0                                    # 10 halting x 5 s
    assert terms["halting_veh"] == 10 and terms["r_vehicle"] == r


def test_reward_excludes_dwelling_buses_but_counts_queued_buses():
    lanes = [lane("A1B1_0", halting=6), lane("C1B1_0", halting=2)]   # incl. 2 halted buses
    buses = [bus("L1_EB_01", "A1B1_0", 170.0, occ=40, at_stop=True),   # dwelling: excluded
             bus("L1_WB_01", "C1B1_0", 20.0, occ=50),                  # queued at the light
             bus("L2_NB_01", "B0B1_0", 90.0, occ=60, speed=8.0)]       # moving: not halting
    s = state(lanes, buses)
    r, terms = rw.reward_vehicle(None, s)
    assert terms["halting_cars"] == 6 and terms["halting_buses"] == 1
    assert r == -(6 + 1) * 5.0
    r, terms = rw.reward_person(None, s)
    assert terms["halting_persons"] == pytest.approx(1.3 * 6 + 50)
    assert terms["halting_bus_persons"] == pytest.approx(50.0)
    assert r == pytest.approx(-(1.3 * 6 + 50) * 5.0)


def test_reward_ignores_buses_off_the_incoming_lanes():
    s = state([lane("A1B1_0", halting=2)], [bus("L1_EB_01", "elsewhere_0", 50.0, occ=40)])
    assert rw.reward_person(None, s)[0] == pytest.approx(-1.3 * 2 * 5.0)


def test_rewards_zero_without_queue_and_never_positive():
    s = state(b1_lanes())
    assert rw.reward_vehicle(None, s)[0] == 0.0 and rw.reward_person(None, s)[0] == 0.0
    s = state(b1_lanes(halting=3), [bus("L1_EB_01", "A1B1_0", 10.0, occ=30)])
    assert rw.reward_vehicle(None, s)[0] < 0 and rw.reward_person(None, s)[0] < 0


# reference rewards (phase 1)
def test_reward_vehicle_diff_is_minus_change_in_waiting():
    prev = state([lane("A1B1_0", waiting=100.0), lane("C1B1_0", waiting=50.0)])
    now = state([lane("A1B1_0", waiting=40.0), lane("C1B1_0", waiting=80.0)])
    r, terms = rw.reward_vehicle_diff(prev, now)
    assert r == pytest.approx(30.0)                      # 150 -> 120 s
    assert terms["wait_veh"] == 120.0


def test_reward_person_diff_weights_cars_and_buses():
    prev = state([lane("A1B1_0", waiting=0.0)])
    now = state([lane("A1B1_0", waiting=50.0)],          # 30 s car + 20 s bus
                [bus("L1_EB_01", "A1B1_0", 50.0, occ=40, waiting=20.0)])
    r, terms = rw.reward_person_diff(prev, now)
    assert terms["wait_person"] == pytest.approx(1.3 * 30.0 + 40 * 20.0)
    assert r == pytest.approx(-(1.3 * 30.0 + 800.0))
