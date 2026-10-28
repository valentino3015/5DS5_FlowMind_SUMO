import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rule_tsp import (EW_PHASE, NS_PHASE, BusView, JunctionView, Params,  # noqa: E402
                      decide, is_late, priority_key)

YELLOW_EW = 1   # phase index of the E-W yellow


def view(phase=EW_PHASE, spent=20.0, remaining=15.0, ext_used=0.0, buses=(), queues=None):
    return JunctionView(tls="B1", t=1000.0, phase=phase, spent=spent, remaining=remaining,
                        ext_used=ext_used, buses=list(buses),
                        queues=queues if queues is not None else {EW_PHASE: 0, NS_PHASE: 0})


def bus(bid="L1_EB_01", phase=EW_PHASE, dist=100.0, speed=10.0, occ=30.0, late=60.0):
    return BusView(id=bid, phase=phase, dist=dist, speed=speed, occupancy=occ, lateness=late)


# ---- lateness -------------------------------------------------------------
def test_is_late():
    assert is_late(34.0, 0.0)
    assert not is_late(-20.0, 0.0)
    assert not is_late(0.0, 0.0)          # strictly greater than the threshold
    assert not is_late(None, 0.0)         # no previous bus on record


def test_cond_refuses_on_time_bus_but_all_serves_it():
    b = bus(late=-20.0, dist=140.0, speed=10.0)      # needs 16 s, only 15 s of green left
    d_c = decide(view(buses=[b]), Params(mode="cond"))
    assert d_c.action == "none"
    assert d_c.refusals[0][1] == "not_late" and "20 s early" in d_c.refusals[0][2]
    d_b = decide(view(buses=[b]), Params(mode="all"))
    assert d_b.action == "extend" and d_b.new_remaining == 16.0


def test_cond_extends_late_bus_with_reason():
    d = decide(view(buses=[bus(dist=140.0, late=34.0)]), Params(mode="cond"))
    assert d.action == "extend" and d.new_remaining == 16.0
    assert d.reason == "extend 1 s for L1_EB_01, 34 s late"


def test_no_request_when_bus_clears():
    d = decide(view(buses=[bus(dist=50.0)]), Params(mode="all"))
    assert d.action == "none" and d.refusals == []


# ---- queue refusal -----------------------------------------------------------
def test_cond_queue_refusal_extension_checks_waiting_approaches():
    b = bus(dist=140.0)
    d = decide(view(buses=[b], queues={EW_PHASE: 0, NS_PHASE: 16}), Params(mode="cond"))
    assert d.action == "none" and d.refusals[0][1] == "queue"
    d = decide(view(buses=[b], queues={EW_PHASE: 99, NS_PHASE: 15}), Params(mode="cond"))
    assert d.action == "extend"                       # 15 is not > 15; EW queue is not losing


def test_cond_queue_refusal_recall_checks_current_green():
    b = bus(bid="L2_NB_03", phase=NS_PHASE, dist=40.0, speed=0.0)
    d = decide(view(buses=[b], queues={EW_PHASE: 20, NS_PHASE: 0}), Params(mode="cond"))
    assert d.action == "none" and d.refusals[0][1] == "queue"
    d = decide(view(buses=[b], queues={EW_PHASE: 5, NS_PHASE: 30}), Params(mode="cond"))
    assert d.action == "recall"


def test_all_ignores_queues():
    d = decide(view(buses=[bus(dist=140.0)], queues={EW_PHASE: 0, NS_PHASE: 99}),
               Params(mode="all"))
    assert d.action == "extend"


# ---- conflict resolution -------------------------------------------------------
def test_cond_conflict_later_bus_wins():
    ew = bus("L1_EB_01", EW_PHASE, dist=140.0, late=30.0)          # wants extension
    ns = bus("L2_NB_02", NS_PHASE, dist=60.0, speed=0.0, late=90.0)  # wants recall
    d = decide(view(buses=[ew, ns]), Params(mode="cond"))
    assert d.action == "recall" and d.bus == "L2_NB_02"
    assert ("L1_EB_01", "conflict") in [(r[0], r[1]) for r in d.refusals]


def test_cond_conflict_tie_higher_occupancy():
    ew = bus("L1_EB_01", EW_PHASE, dist=140.0, late=30.0, occ=55)
    ns = bus("L2_NB_02", NS_PHASE, dist=60.0, speed=0.0, late=30.0, occ=25)
    d = decide(view(buses=[ew, ns]), Params(mode="cond"))
    assert d.action == "extend" and d.bus == "L1_EB_01"
    assert priority_key(ew) > priority_key(ns)


def test_all_conflict_prefers_current_green():
    ew = bus("L1_EB_01", EW_PHASE, dist=140.0, late=-50.0)
    ns = bus("L2_NB_02", NS_PHASE, dist=10.0, speed=0.0, late=200.0)
    d = decide(view(buses=[ew, ns]), Params(mode="all"))
    assert d.action == "extend" and d.bus == "L1_EB_01"


# ---- timing limits -------------------------------------------------------------
def test_recall_never_below_min_green():
    ns = bus("L2_NB_02", NS_PHASE, dist=30.0, speed=0.0)
    d = decide(view(spent=4.0, remaining=31.0, buses=[ns]), Params(mode="all"))
    assert d.action == "recall" and d.new_remaining == 6.0      # 4 + 6 = min green 10
    d = decide(view(spent=25.0, remaining=10.0, buses=[ns]), Params(mode="all"))
    assert d.action == "recall" and d.new_remaining == 0.0      # already past min green


def test_extension_respects_budget_and_max_green():
    far = bus(dist=145.0, speed=0.0)                             # needs 31 s
    d = decide(view(remaining=5.0, buses=[far]), Params(mode="all"))
    assert d.action == "none" and d.refusals[0][1] == "budget"
    b = bus(dist=140.0)                                          # needs +1 s
    d = decide(view(remaining=15.0, ext_used=15.0, buses=[b]), Params(mode="all"))
    assert d.action == "none" and d.refusals[0][1] == "budget"
    d = decide(view(spent=45.0, remaining=15.0, buses=[b]), Params(mode="all"))
    assert d.action == "none" and d.refusals[0][1] == "max_green"


def test_yellow_and_all_red_never_touched():
    ns = bus("L2_NB_02", NS_PHASE, dist=30.0, speed=0.0)
    for phase in (1, 2, 4, 5):
        assert decide(view(phase=phase, buses=[ns, bus()]), Params(mode="all")).action == "none"


def test_random_views_respect_min_max_green_and_budget():
    rng = random.Random(0)
    for _ in range(5000):
        p = Params(mode=rng.choice(["all", "cond"]), queue_thr=rng.randint(0, 30))
        phase = rng.choice([EW_PHASE, NS_PHASE, 1, 2, 4, 5])
        spent = rng.uniform(0, 60)
        remaining = rng.uniform(0, max(0.0, 60 - spent))
        ext_used = rng.uniform(0, p.max_ext)
        buses = [bus(f"L{rng.randint(1, 2)}_X_{i}", rng.choice([EW_PHASE, NS_PHASE]),
                     rng.uniform(0, 150), rng.uniform(0, 14), rng.uniform(20, 60),
                     rng.choice([None, rng.uniform(-200, 200)])) for i in range(rng.randint(0, 4))]
        v = view(phase, spent, remaining, ext_used, buses,
                 {EW_PHASE: rng.randint(0, 40), NS_PHASE: rng.randint(0, 40)})
        d = decide(v, p)
        if d.action == "none":
            continue
        assert phase in (EW_PHASE, NS_PHASE)
        green = spent + d.new_remaining
        if d.action == "extend":
            assert green <= p.max_green + 1e-9
            assert d.new_remaining - remaining <= p.max_ext - ext_used + 1e-9
        else:
            assert green >= p.min_green - 1e-9 or d.new_remaining == remaining
            assert d.new_remaining < remaining
