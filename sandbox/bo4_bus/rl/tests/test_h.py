"""Agent H: feasible actions (pure), mask, observation extras, naming."""
import os
import sys

import numpy as np
import pytest

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
import h_agent as h  # noqa: E402
import observation as ob  # noqa: E402
from rule_tsp import EW_PHASE, NS_PHASE, BusView, Decision, JunctionView, Params  # noqa: E402

P = Params(mode="cond", late_thr=0.0, queue_thr=15, max_ext=20.0)


def view(phase=EW_PHASE, spent=20.0, remaining=15.0, ext_used=0.0, queues=None):
    return JunctionView(tls="B1", t=1000.0, phase=phase, spent=spent, remaining=remaining,
                        ext_used=ext_used, queues=queues or {EW_PHASE: 3, NS_PHASE: 3})


def bus(phase=EW_PHASE, dist=60.0, speed=10.0, late=-40.0):
    return BusView(id="L1_EB_01", phase=phase, dist=dist, speed=speed, occupancy=30.0,
                   lateness=late)


def test_follow_always_and_nothing_outside_green():
    assert set(h.h_options(view(phase=1), bus(), P)) == {h.FOLLOW}        # yellow
    assert h.FOLLOW in h.h_options(view(), bus(), P)


def test_serve_on_green_extends_so_the_bus_clears():
    b = bus(dist=140.0, speed=10.0)                     # needs 16 s, 3 s left
    o = h.h_options(view(spent=30.0, remaining=3.0), b, P)
    kind, new_rem, ext = o[h.SERVE]
    assert kind == "extend" and new_rem == pytest.approx(16.0) and ext == pytest.approx(13.0)


def test_serve_on_red_recalls_and_hold_on_red_keeps_red():
    v = view(phase=NS_PHASE, spent=20.0, remaining=10.0)
    o = h.h_options(v, bus(phase=EW_PHASE), P)
    assert o[h.SERVE][:2] == ("recall", 0.0)            # N-S already past its 10 s minimum
    assert o[h.HOLD] == ("hold_red", 10.0, 0.0)          # already runs past the next tick
    o = h.h_options(view(phase=NS_PHASE, spent=20.0, remaining=2.0), bus(), P)
    assert o[h.HOLD] == ("hold_red", 6.0, 4.0)           # keep red until the next tick


def test_hold_on_green_cuts_to_min_green():
    o = h.h_options(view(spent=4.0, remaining=31.0), bus(), P)
    assert o[h.HOLD] == ("cut", 6.0, 0.0)                # green ends at 10 s
    o = h.h_options(view(spent=12.0, remaining=3.0), bus(dist=10.0), P)
    assert o[h.HOLD] == ("cut", 0.0, 0.0)                # ends now (own yellow)


def test_limits_mask_serve_and_hold():
    # extension budget used up -> no serve-extend, no hold-extend
    o = h.h_options(view(spent=30.0, remaining=3.0, ext_used=20.0), bus(dist=140.0), P)
    assert h.SERVE not in o
    o = h.h_options(view(phase=NS_PHASE, spent=20.0, remaining=2.0, ext_used=20.0), bus(), P)
    assert h.HOLD not in o
    # max green: N-S at 56 s + 2 s + 4 s > 60 s
    o = h.h_options(view(phase=NS_PHASE, spent=56.0, remaining=2.0), bus(), P)
    assert h.HOLD not in o
    # C's queue guard: cutting the bus's own green with 16 halting there is refused
    o = h.h_options(view(spent=4.0, remaining=31.0, queues={EW_PHASE: 16, NS_PHASE: 0}), bus(), P)
    assert h.HOLD not in o
    # green already ending: nothing to cut
    assert h.HOLD not in h.h_options(view(spent=4.0, remaining=5.0), bus(), P)


def test_every_option_respects_green_bounds():
    for phase in (EW_PHASE, NS_PHASE):
        for spent in range(0, 61, 3):
            for rem in range(0, 61 - spent, 4):
                for bphase in (EW_PHASE, NS_PHASE):
                    for dist in (5.0, 80.0, 149.0):
                        v = view(phase=phase, spent=float(spent), remaining=float(rem))
                        for kind, new_rem, ext in h.h_options(v, bus(bphase, dist), P).values():
                            if new_rem is None:
                                continue
                            total = spent + new_rem
                            assert total <= P.max_green + 1e-9, (kind, spent, rem)
                            assert new_rem <= 0 or total >= min(P.min_green, spent + rem) - 1e-9
                            assert ext <= P.max_ext + 1e-9


def test_mask_and_extras():
    m = h.action_mask({h.FOLLOW: None, h.HOLD: None})
    assert list(m) == [True, False, True]
    x = h.extras(view(ext_used=5.0), bus(late=-400.0, dist=300.0), P, True,
                 Decision(action="extend"))
    assert list(x) == pytest.approx([-2.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.75])
    assert h.OBS_SIZE_H == 43 == h.LOW_H.size == h.HIGH_H.size
    assert ob.version_of_size(43) == "h"
    assert np.all(x >= h.LOW_H[36:]) and np.all(x <= h.HIGH_H[36:])


def test_run_name_h():
    import train_ppo as tp
    assert tp.run_name("regular_h", 1.0, "v2", 0, 200000, None, 0.5, True, 4, 1.0) == \
        "ppo_H_h4_d1_c0.5_s0_200k"
