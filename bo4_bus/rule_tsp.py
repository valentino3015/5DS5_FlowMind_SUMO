"""
Rule-based transit signal priority (TSP) for the BO4 3x3 grid.

One controller object per signalized junction, plugged into Grid_Demo.py's
`controllers[tls_id].act(t, sim)` hook.

Structure (so the logic can be ported to the team's JunctionState contract):
    sense(sim, geo, ...)   the ONLY place that calls SUMO; returns a JunctionView
                           made of plain data (dataclasses, floats, strings)
    decide(view, params)   pure function: JunctionView -> Decision (no SUMO)
    TspController.act()    glue: sense -> lateness bookkeeping -> decide -> apply

Signal program (written by nets/grid/build_grid.py), identical at every junction:
    0 E-W green 35 | 1 yellow 3 | 2 all-red 2 | 3 N-S green 30 | 4 yellow 3 | 5 all-red 2
Priority only changes the remaining time of the CURRENT green phase:
    green extension : remaining += need          (bus on the green approach would not clear)
    early recall    : remaining  = max(0, MIN_GREEN - spent)   (bus waits on red)
Yellow and all-red are never touched, the program then continues with its
normal durations ("return to the normal fixed-time cycle"). Guarantees:
    green >= MIN_GREEN, green <= MAX_GREEN, extensions <= max_ext per cycle.

Modes:
    "all"  (controller B, unconditional): every detected bus may get priority.
           Conflicting requests: the bus served by the current green wins (an
           extension serves a bus immediately), then the nearest bus.
    "cond" (controller C, conditional):
           - only late buses: lateness = t - (last passage of a bus of the same
             line direction at this junction) - H; priority if lateness > late_thr;
             a bus with no previous passage on record is not prioritized;
           - refused if the max halting queue over the approaches that would lose
             green exceeds queue_thr;
           - conflicting requests: the later bus wins (tie: higher occupancy).
"""
import math
import os
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "nets", "grid"))
from build_grid import ALL_RED, LINES, MAX_GREEN, MIN_GREEN, TLS_IDS, YELLOW  # noqa: E402

EW_PHASE, NS_PHASE = 0, 3            # green phase indices of every grid junction
GREEN_PHASES = (EW_PHASE, NS_PHASE)
N_PHASES = 6
CONFLICT = {EW_PHASE: NS_PHASE, NS_PHASE: EW_PHASE}

# defaults (all overridable from the CLI, see add_args)
DETECT_DIST = 150.0      # m before the stop line
MAX_EXT = 15.0           # s of green extension per junction per cycle
HEADWAY = 180.0          # s, scheduled headway H of every line
LATE_THR = 0.0           # s, controller C prioritizes lateness > LATE_THR
QUEUE_THR = 15           # veh, controller C refuses if a losing approach queue > QUEUE_THR
V_MIN_EST = 5.0          # m/s, speed assumed for a slow/stopped bus when estimating clearing
CLEAR_MARGIN = 2.0       # s, safety margin for the bus to clear the stop line

BUS_PREFIXES = tuple(f"{d}_" for d in LINES)   # bus ids are "<line dir>_<k>"


def line_of(bus_id):
    """'L1_EB_12' -> 'L1_EB'."""
    return bus_id.rsplit("_", 1)[0]


# --------------------------------------------------------------------------
# plain data
# --------------------------------------------------------------------------
@dataclass
class Params:
    mode: str = "all"                # "all" (B) or "cond" (C)
    detect_dist: float = DETECT_DIST
    max_ext: float = MAX_EXT
    headway: float = HEADWAY
    late_thr: float = LATE_THR
    queue_thr: int = QUEUE_THR
    min_green: float = MIN_GREEN
    max_green: float = MAX_GREEN
    v_min_est: float = V_MIN_EST
    clear_margin: float = CLEAR_MARGIN


@dataclass
class BusView:
    id: str
    phase: int                  # green phase serving its approach
    dist: float                 # m to the stop line
    speed: float                # m/s
    occupancy: float = 0.0
    lateness: float = None      # s, None = no previous passage on record


@dataclass
class JunctionView:
    tls: str
    t: float
    phase: int
    spent: float                # s already spent in the current phase
    remaining: float            # s until the current phase ends
    ext_used: float             # s of extension already given in this cycle
    buses: list = field(default_factory=list)
    queues: dict = field(default_factory=dict)   # green phase -> max halting over its approaches


@dataclass
class Decision:
    action: str = "none"        # "none" | "extend" | "recall"
    new_remaining: float = None # remaining time to set on the current green phase
    bus: str = None
    reason: str = ""
    code: str = ""
    refusals: list = field(default_factory=list)   # [(bus_id, code, reason)]


# --------------------------------------------------------------------------
# pure decision logic
# --------------------------------------------------------------------------
def time_to_clear(bus, p):
    """Estimated time (s) for the bus to pass the stop line."""
    return bus.dist / max(bus.speed, p.v_min_est) + p.clear_margin


def is_late(lateness, late_thr):
    """Controller C eligibility: lateness known and above the threshold."""
    return lateness is not None and lateness > late_thr


def priority_key(bus):
    """Controller C conflict order: later first, then higher occupancy."""
    return (bus.lateness if bus.lateness is not None else -math.inf, bus.occupancy)


def _fmt_late(bus):
    if bus.lateness is None:
        return "no previous bus on record"
    return f"{bus.lateness:.0f} s late" if bus.lateness >= 0 else f"{-bus.lateness:.0f} s early"


def request_for(bus, v, p):
    """What this bus would need now, ignoring eligibility/queues/conflicts.
    Returns (action, new_remaining, code, reason); action None = nothing to do."""
    if v.phase == bus.phase:                                   # bus approach is green
        need = math.ceil(time_to_clear(bus, p) - v.remaining)
        if need <= 0:
            return None, None, "clears", "clears in current green"
        if need > p.max_ext - v.ext_used:
            return "refuse", None, "budget", (f"needs +{need} s, only "
                                              f"{p.max_ext - v.ext_used:.0f} s extension left")
        if v.spent + v.remaining + need > p.max_green:
            return "refuse", None, "max_green", f"+{need} s would exceed max green {p.max_green:.0f} s"
        return "extend", v.remaining + need, "extend", f"extend {need} s"
    if v.phase == CONFLICT[bus.phase]:                         # conflicting green running
        new_rem = max(0.0, p.min_green - v.spent)
        if new_rem >= v.remaining:
            return None, None, "ending", "conflicting green already ending"
        return "recall", new_rem, "recall", (f"cut green to {v.spent + new_rem:.0f} s "
                                             f"(min {p.min_green:.0f} s)")
    return None, None, "transition", "yellow/all-red running"   # never touched


def decide(v, p):
    """Pure TSP decision for one junction. Never returns a green below p.min_green,
    above p.max_green, or extensions beyond p.max_ext per cycle."""
    d = Decision()
    cands = []
    for bus in v.buses:
        action, new_rem, code, reason = request_for(bus, v, p)
        if action is None:
            continue
        if action == "refuse":
            d.refusals.append((bus.id, code, f"refuse {bus.id}: {reason}"))
            continue
        if p.mode == "cond":
            if not is_late(bus.lateness, p.late_thr):
                d.refusals.append((bus.id, "not_late", f"refuse {bus.id}: {_fmt_late(bus)}"))
                continue
            losing = v.phase if action == "recall" else CONFLICT[v.phase]
            q = v.queues.get(losing, 0)
            if q > p.queue_thr:
                d.refusals.append((bus.id, "queue", f"refuse {bus.id}: queue {q} veh on losing "
                                                    f"approaches > {p.queue_thr}"))
                continue
        cands.append((bus, action, new_rem, reason))
    if not cands:
        return d

    if p.mode == "cond":
        winner = max(cands, key=lambda c: priority_key(c[0]))
    else:  # unconditional: extension (served now) before recall, then nearest
        winner = min(cands, key=lambda c: (c[1] != "extend", c[0].dist))
    bus, action, new_rem, reason = winner
    # several buses on the winning side: serve the most demanding one that fits
    same = [c for c in cands if c[1] == action]
    if action == "extend":
        bus, action, new_rem, reason = max(same, key=lambda c: c[2])
    for c in cands:
        if c[1] != action:
            d.refusals.append((c[0].id, "conflict",
                               f"refuse {c[0].id}: conflicting request, {bus.id} served"))

    # hard guarantees (defensive; request_for already respects them)
    total = v.spent + new_rem
    assert action == "recall" or total <= p.max_green + 1e-9, total
    assert total >= min(p.min_green, v.spent + v.remaining) - 1e-9, total
    d.action, d.new_remaining, d.bus, d.code = action, new_rem, bus.id, action
    d.reason = f"{reason} for {bus.id}, {_fmt_late(bus)}"
    return d


# --------------------------------------------------------------------------
# SUMO side
# --------------------------------------------------------------------------
def junction_geometry(net):
    """{tls: [{"edges": [main, piece], "len": {edge: lane length}, "phase": green phase}]}"""
    geo = {}
    for tid in TLS_IDS:
        appr = []
        pieces = {lane.getEdge().getID() for lane, _, _ in net.getTLS(tid).getConnections()}
        for piece in sorted(pieces):
            main = piece.replace(".250.00", "")
            e = net.getEdge(main)
            (x0, y0), (x1, y1) = e.getFromNode().getCoord(), net.getEdge(piece).getToNode().getCoord()
            phase = EW_PHASE if abs(x1 - x0) > abs(y1 - y0) else NS_PHASE
            appr.append({"edges": [main, piece], "phase": phase,
                         "len": {main: e.getLane(0).getLength(),
                                 piece: net.getEdge(piece).getLane(0).getLength()}})
        geo[tid] = appr
    return geo


def sense(sim, tls, appr, t, p, occupancy):
    """Read the junction state from SUMO into a JunctionView (lateness and ext_used are
    filled in by the controller). Buses dwelling at, or about to stop at, a bus stop
    before the stop line are not detected."""
    phase = sim.trafficlight.getPhase(tls)
    v = JunctionView(tls=tls, t=t, phase=phase,
                     spent=sim.trafficlight.getSpentDuration(tls),
                     remaining=sim.trafficlight.getNextSwitch(tls) - t, ext_used=0.0)
    for a in appr:
        main, piece = a["edges"]
        for edge in (main, piece):
            for veh in sim.edge.getLastStepVehicleIDs(edge):
                if not veh.startswith(BUS_PREFIXES):
                    continue
                pos = sim.vehicle.getLanePosition(veh)
                dist = a["len"][piece] - pos + (a["len"][main] if edge == main else 0.0)
                if dist > p.detect_dist or sim.vehicle.isAtBusStop(veh):
                    continue
                nxt = sim.vehicle.getStops(veh, 1)
                if nxt and nxt[0].lane.rsplit("_", 1)[0] == edge and nxt[0].endPos > pos:
                    continue                      # will dwell before reaching the stop line
                if veh not in occupancy:
                    occupancy[veh] = float(sim.vehicle.getParameter(veh, "occupancy") or 0)
                v.buses.append(BusView(id=veh, phase=a["phase"], dist=dist,
                                       speed=sim.vehicle.getSpeed(veh), occupancy=occupancy[veh]))
    if v.buses:
        for g in GREEN_PHASES:
            v.queues[g] = max(sum(sim.edge.getLastStepHaltingNumber(e) for e in a["edges"])
                              for a in appr if a["phase"] == g)
    return v


class TspController:
    """Per-junction controller: act(t, sim) senses, decides and applies."""

    def __init__(self, tls_id, appr, params, occupancy_cache):
        self.tls_id = tls_id
        self.appr = appr
        self.p = params
        self.occ = occupancy_cache
        self.log = []              # one row per decision event (deduplicated)
        self.outcomes = []         # one row per (bus, junction) request
        self._seen = set()         # log dedup keys
        self._pending = {}         # bus -> {"granted": bool, "code": last refusal code}
        self._prev_buses = set()   # buses detected last step
        self._last_pass = {}       # line dir -> time a bus of it last passed this junction
        self._ext_used = 0.0
        self._last_phase = None

    def reset(self, sim):
        pass

    def _record(self, t, bus, action, code, reason, lateness, seconds=None):
        key = (bus, action, code)
        if key in self._seen:
            return
        self._seen.add(key)
        self.log.append({"t": t, "tls": self.tls_id, "bus": bus, "line": line_of(bus),
                         "action": action, "code": code, "seconds": seconds,
                         "lateness": lateness, "reason": reason})

    def act(self, t, sim):
        v = sense(sim, self.tls_id, self.appr, t, self.p, self.occ)

        # cycle bookkeeping: a new cycle starts when the E-W green starts
        if v.phase == EW_PHASE and self._last_phase not in (None, EW_PHASE):
            self._ext_used = 0.0
        self._last_phase = v.phase
        v.ext_used = self._ext_used

        # passages: buses that left the detection zone towards the junction
        now = {b.id for b in v.buses}
        for bus in self._prev_buses - now:      # crossed the stop line (or teleported)
            self._last_pass[line_of(bus)] = t
            st = self._pending.pop(bus, None)
            if st is not None:
                self.outcomes.append({"t": t, "tls": self.tls_id, "bus": bus, "line": line_of(bus),
                                      "outcome": "granted" if st["granted"] else "refused",
                                      "code": st["code"]})
        self._prev_buses = now
        for b in v.buses:
            last = self._last_pass.get(line_of(b.id))
            b.lateness = None if last is None else t - last - self.p.headway

        d = decide(v, self.p)
        lat = {b.id: b.lateness for b in v.buses}
        for bus, code, reason in d.refusals:
            st = self._pending.setdefault(bus, {"granted": False, "code": None})
            st["code"] = code
            self._record(t, bus, "refuse", code, reason, lat.get(bus))
        if d.action == "none":
            return
        st = self._pending.setdefault(d.bus, {"granted": False, "code": None})
        st["granted"] = True
        seconds = d.new_remaining - v.remaining
        if d.action == "extend":
            self._ext_used += seconds
        if d.new_remaining <= 0:
            sim.trafficlight.setPhase(self.tls_id, (v.phase + 1) % N_PHASES)   # its own yellow
        else:
            sim.trafficlight.setPhaseDuration(self.tls_id, d.new_remaining)
        self._record(t, d.bus, d.action, d.code, d.reason, lat.get(d.bus), round(seconds, 1))


def make_controllers(mode, net, params, fixed_cls):
    """{tls: controller}. mode: "fixed" -> fixed_cls(tls); "tsp_all"/"tsp_cond" -> TspController."""
    if mode == "fixed":
        return {tid: fixed_cls(tid) for tid in TLS_IDS}
    params.mode = {"tsp_all": "all", "tsp_cond": "cond"}[mode]
    geo = junction_geometry(net)
    occ = {}
    return {tid: TspController(tid, geo[tid], params, occ) for tid in TLS_IDS}


def add_args(ap):
    ap.add_argument("--detect-dist", type=float, default=DETECT_DIST,
                    help="bus detection distance before the stop line [m]")
    ap.add_argument("--max-ext", type=float, default=MAX_EXT,
                    help="max green extension per junction per cycle [s]")
    ap.add_argument("--late-thr", type=float, default=LATE_THR,
                    help="controller C: prioritize buses later than this [s]")
    ap.add_argument("--queue-thr", type=int, default=QUEUE_THR,
                    help="controller C: refuse if a losing approach queue exceeds this [veh]")
    return ap


def params_from_args(args):
    return Params(detect_dist=args.detect_dist, max_ext=args.max_ext, headway=args.headway,
                  late_thr=args.late_thr, queue_thr=args.queue_thr)


assert YELLOW + ALL_RED > 0 and MIN_GREEN < MAX_GREEN
