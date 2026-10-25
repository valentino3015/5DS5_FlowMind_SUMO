# BO4 bus scenario – assumptions (step 1: fixed-time baseline)

## Network (`nets/build_net.py`)
- Arterial W–E: 600 m each side, 2 lanes/direction. Cross street N–S: 300 m each side, 1 lane/direction.
  Speed limit **13.89 m/s on both streets** (the cross-street speed was not specified).
- netconvert trims each approach to **592.8 m** (arterial) because of the junction area.
- Signal `C`, static, cycle **75 s** (team defaults): arterial green 40 / yellow 3 / all-red 2 /
  cross green 25 / yellow 3 / all-red 2. No protected left phase.
  - Green phases carry min green 10 s / max green 60 s (`minDur` / `maxDur` in `bus.tll.xml`, and `MIN_GREEN` / `MAX_GREEN`
    in `build_net.py`). The static program ignores them, and netconvert drops them from `bus.net.xml`.
    Adaptive controllers must enforce them. Left turns are `g`: permitted, unprotected,
  yielding to opposing traffic. Right turns and through movements are `G`. No pedestrians.
- The phase states are derived automatically from the connection directions (sumolib) and
  embedded into `bus.net.xml` in a second netconvert pass.
- Bus stops sit on the rightmost lane (`_0`) and are 30 m long, centred 150 m from the junction.
  Upstream stops span 427.8–457.8 m on `W2C_0` / `E2C_0`; downstream stops span 135–165 m on `C2E_0` / `C2W_0`.
- On the arterial, lane 0 carries right + through and lane 1 carries through + left.
  On the cross street, the single lane carries all three movements.

## Demand (`gen_routes.py`)
- Cars are written as explicit vehicles with Poisson arrivals (exponential gaps, numpy seeded with `--seed`),
  so a seed fully defines the demand. Generated departures cover [0, 4200) s.
- **Default demand used for the baseline (changed from the brief, see "Calibration"):**
  | | flow (veh/h per direction) | left | right | through |
  |---|---|---|---|---|
  | arterial | **1100** (brief: ~800) | 5 % | 10 % | 85 % |
  | cross    | 450                     | 10 % | 15 % | 75 % |
- Car vType: length 5 m, accel 2.6, decel 4.5, sigma 0.5 (SUMO Krauss defaults).
  **Car occupancy = 1.3** (constant `CAR_OCCUPANCY`).
- Bus vType: vClass `bus`, length 12 m, accel 1.2 m/s², decel 4.0 m/s², maxSpeed 13.89 m/s,
  speedFactor 1 (no speeding, so free-flow speed = speed limit).
- Bus line L1: EB route `W2C C2E`, WB route `E2C C2W`. Param `line` = `L1_EB` / `L1_WB`.
  Bus `occupancy` ~ uniform integer in [20, 60].
- Dispatch: `t_k = 60 + k·H + clip(N(0, 45 s), ±0.45·H)`, with H = 240 s.
  Clipping to ±0.45 H guarantees the dispatch order is preserved.
- Dwell: `duration = 10 + 2.5·B` with `B ~ Poisson(λ·gap)` and `λ = (25 − 10)/(2.5·H) = 0.025 pax/s`,
  so the mean dwell is 25 s at the nominal headway.
- **Open-loop dwell:** the dwell is written into the route file before the simulation runs, so `gap` is estimated:
  - At the upstream stop, `gap` is the dispatch gap. Every bus has the same free-flow running time from dispatch to this stop.
  - At the downstream stop, `gap` is the dispatch gap plus the difference in upstream dwell.
  - Signal delay at the junction is **not** fed back into the downstream dwell.
  - A closed-loop dwell set through TraCI on arrival would be more faithful. It is left for a later step.

## Measurement (`Bus_Demo.py`, `bus_metrics.py`)
- **Evaluation seeds are 10000–10004** (team rule: evaluation seeds start at 10000). Seeds below 10000
  were only used for the calibration sweeps.
- Warm-up runs from 0 to 600 s; measurement runs from 600 to 4200 s. The SUMO `--seed` is the same as the demand seed.
  `--time-to-teleport` is 300 s (the SUMO default).
- **Bus junction delay** = (`started` at the downstream stop − `ended` at the upstream stop) − free-flow time.
  - Free-flow time = distance / 13.89 m/s. The distance runs from the upstream stop endPos to the downstream stop endPos:
    lane remainder + internal junction lane + downstream endPos. A stopped bus has its front at endPos.
  - This is pure distance/speed with no accel/decel term, so a bus that never stops for the signal still shows a few
    seconds of "delay" from leaving and re-entering a stop.
  - A bus is counted if it left the upstream stop at or after 600 s and reached the downstream stop before 4200 s.
- **Headways** come from `started` in the stop-output, using stop arrivals in [600, 4200) s per stop.
  **CV** = sample std (ddof = 1) / mean.
  **Bunching** = headway < 0.5·H (120 s).
- **"Waits more than one cycle"** is proxied by junction delay > cycle length (read from the net, 75 s).
- **Car delay** = mean `timeLoss` of cars that departed in the window. Unfinished trips are included
  (`--tripinfo-output.write-unfinished`).
  - `timeLoss` excludes insertion delay, so the mean `departDelay` is reported separately (`car_depart_delay`).
  - Vehicles that are still un-inserted at 4200 s produce no tripinfo. They are counted in `pending_insert`.
- **Person delay** = Σ timeLoss × occupancy over buses (their own `occupancy`) and cars (1.3), for trips departed
  in the window. It is reported in person-hours.
- **Cross queue** = max over seconds of the halting vehicles (speed < 0.1 m/s) on `N2C` / `S2C`.
  A value of 38 is roughly a full 300 m link.
- **Arterial queue** (used for the congestion check) = per second, the farther lane's contiguous queue from the
  stop line. A vehicle counts as queued below 5 km/h, with gaps of at most 25 m between queued vehicles.
  `art_q_at_stop_*` is the share of seconds this queue is ≥ 135 m, i.e. it reaches the upstream stop zone.

## Calibration (what was changed and why)
Calibration was done with the earlier 71 s cycle (no all-red) on seeds 1–5. The demand was **not** re-tuned
after switching to the 75 s team timing.
- With the brief's demand (arterial 800 veh/h, my initial turning shares 10/10 arterial and 20/20 cross), seed 1 showed:
  - The arterial queue maxed out at about 95 m and never reached the upstream stop.
  - No bus waited more than one cycle.
  - The cross street jammed. Its single shared lane, with 20 % unprotected lefts, filled the 300 m link and
    107 vehicles were left waiting for insertion.
- The arterial capacity is set by the **unprotected lefts in shared lane 1**, which face 2-lane opposing flow.
  Each direction discharges only about 1000–1150 veh/h, and lane 1 queues while lane 0 (the bus lane) stays almost empty.
- The arterial demand was therefore raised to 1100 veh/h, with arterial lefts at 5 %.
  Cross-street turning was reduced to 10 % left / 15 % right.
- Swept alternatives (seeds 1–2):
  - 1000 veh/h with 5 % left: the queue rarely reaches the stop and no bus waits more than a cycle.
  - 1000 veh/h with 10 % left: the queue reaches the stop but only in lane 1, and no bus waits more than a cycle.
  - 1200 veh/h with 5 % left: the queue spills back to the 593 m link entry.
- Dispatch jitter (σ = 45 s) and dwell parameters were kept at the brief's values.
  Bunching appears in the baseline, but it is infrequent (see the results).

---

# Grid scenario (step 1b): 3x3 grid, `nets/grid/`, `gen_grid_routes.py`, `Grid_Demo.py`
The single-junction scenario above is unchanged. Evaluation seeds are still 10000–10004, and the team timing still applies.

## Network (`nets/grid/build_grid.py`, netgenerate)
- 3x3 junctions `A0..C2` (letter = column, digit = row), 300 m blocks, 300 m fringe links, 2 lanes per direction, 13.89 m/s.
- Signals are at the 9 inner junctions only. The fringe nodes are dead ends.
- **Left-turn lanes:** netgenerate's `--turn-lanes 1 --turn-lanes.length 50` splits every approach at 250 m.
  - Edge `XY` is the 2-lane part; `XY.250.00` is the last ~50 m with 3 lanes, lane 2 being the left-turn lane.
  - Approach metrics treat `XY` + `XY.250.00` as one approach named `XY`.
- **Fixed-time program** at every junction, offset 0: E-W 35 / 3 / 2, N-S 30 / 3 / 2, cycle 75 s.
  - Lefts are permitted and unprotected (`g`) from their own lane.
  - Greens carry minDur 10 / maxDur 60 in `grid.tll.xml`. netconvert drops them from the static program in the net.

## Bus lines and stops
- L1 runs E-W along the middle row: `L1_EB` (left1→A1→B1→C1→right1) and `L1_WB`.
  L2 runs N-S along the middle column: `L2_NB` (bottom1→B0→B1→B2→top1) and `L2_SB`.
  The two lines cross at `B1`.
- Each direction has 4 stops, all 30 m long on lane 0:
  - `_s0` on the entry link, spanning 135–165 m from the fringe node.
  - `_s1`, `_s2`, `_s3` far-side, spanning 85–115 m after each of the 3 junctions. `_s3` is on the exit link.
  - Every segment `s_k → s_{k+1}` therefore crosses exactly one junction.
- Headway H = 180 s per line direction. Dispatch is `t_k = 30 + k·H + clip(N(0, σ), ±0.45 H)`, with σ = 60 s by default (`--dispatch-sigma`).
- Occupancy ~ U{20..60} per bus. The `line` param is `L1_EB` / `L1_WB` / `L2_NB` / `L2_SB`.

## Closed-loop dwell (every stop, `Grid_Demo.py`)
- Stops in the route file have a placeholder duration of 3600 s. The real dwell is set during the simulation.
- On each TraCI stop-start event of a bus at time t:
  - `gap = t − (arrival of the previous same-direction bus at that stop)`. The first bus at a stop uses gap = H.
  - `B ~ Poisson(λ·gap)` with `λ = (25 − 10)/(2.5·H) = 0.0333 pax/s` for H = 180 s.
  - `dwell = 10 + 2.5·B`, which gives a mean of 25 s at gap = H.
- The bus is released with `vehicle.resume()` once the dwell has elapsed.
- The dwell RNG is `default_rng([seed, 7])`, independent of the demand RNG.
- The stop-start event becomes visible one step after the stop begins, and the release is shifted by that step.
  - Checked on a calibration run: stop-output `ended − started` was drawn dwell + 1.0–1.5 s before the fix (the remainder is the 1 s step).
  - Correlation(gap, dwell) = 0.67, mean dwell 24.7 s.
- Two buses can share a stop (30 m fits two 12 m buses), and a bus may overtake a dwelling bus on lane 1.

## Car demand (`gen_grid_routes.py`)
- Each of the 12 fringe entry links has its own seeded Poisson process at `--entry-flow` veh/h. **Calibrated value: 675.**
- Destination: the exit straight across with probability 0.6 (`--straight-share`). Otherwise one of the 10 other exits uniformly; no U-turn exit.
- Cars are written as `<trip>`s, so SUMO assigns the fastest static route at insertion. Turning comes from the OD pattern.
- `--peak-factor/--peak-start/--peak-duration/--peak-period` add optional demand peaks (non-homogeneous Poisson by thinning).
  They are **off** (factor 1) in the baseline; see calibration.

## Grid metrics (`Grid_Demo.py`, `bus_metrics.py`)
- Warm-up runs from 0 to 600 s; measurement runs from 600 to 7800 s. libsumo is used when headless and traci with `--gui`
  (official results: headless libsumo only, see the end of this file).
  `--time-to-teleport` is 300 s.
- **Per line direction and stop:** headway CV and bunching (< 0.5 H = 90 s) at each stop, in line order.
  - `cv_slope` = least-squares slope of CV vs stop index.
  - `bunch_<dir>` is summed over the 4 stops, so a pair of buses that stays bunched counts once per stop.
- **Bus delay per junction** = `started(s_{k+1}) − ended(s_k)` − free-flow time.
  - Free-flow time = route distance between the stops' endPos (including internal junction lanes) / 13.89 m/s.
  - There is no accel/decel term, so about 10 s of the delay is the cost of stopping.
- **End-to-end bus travel time** = tripinfo `arrival − depart`, for buses that departed and arrived inside the window.
- **Car delay** = mean of `timeLoss + departDelay` over cars that departed in the window, unfinished included.
  `car_timeloss` and `car_depart_delay` are also reported separately.
- **Person delay** = Σ (timeLoss + departDelay) × occupancy over buses and cars (1.3), in person-hours.
- **Queues are sampled every 5 s:**
  - Max halting vehicles per signal approach, in `<tag>_queues.csv`.
  - **Spillback:** a halting vehicle within 20 m of an approach link's upstream end. `spill_longest_s` is the longest continuous spillback on any approach.
  - **Queue reaches stop** (`stopq_reach_*`, **diagnostic only, not a target**; see "Dropped target" below):
    for stops with a signal downstream on the same link (`s0`–`s2`), the contiguous lane-0 queue from the stop line
    (vehicles < 5 km/h, gaps ≤ 25 m) reaches the stop's downstream end.
  - `never_inserted` = vehicles still waiting for insertion at 7800 s.

## Calibration of the grid car demand
All runs below use straight share 0.6 unless noted. "Short" = 600 + 1800 s; "full" = 600 + 7200 s.

| entry flow (veh/h) | run | never inserted | longest spillback | teleports | queue reaches a stop (max share) |
|---|---|---|---|---|---|
| 650 | short, seed 1 | 0 | 0 s | 0 | 0 |
| 700 | full, seeds 1–2 | 0 / 0 | 5 / 0 s | 0 | 0 |
| 725 | full, seeds 1–2 | 1 / 0 | 0 s | 0 | 0 |
| 750 | full, seeds 1–2 | 517 / 1614 | 305 / 930 s | 2 / 94 | 0.07 / 0.24 |
| 800 / 850 (straight 0.8) | full, seed 1 | 0 / 4050 | 60 / 1355 s | 0 / 191 | 0 / 0.29 |
| 700 + 20 % peak for 15 min/h | full, seeds 1–2 | 1710 / 179 | 705 / 245 s | 44 / 0 | 0.34 / 0.04 |
| 700 | full, eval seeds 10000–10004 | 0, 0, **303**, 0, 0 | 0–60 s, **645 s** | 0, 0, **25**, 0, 0 | ≤ 0.10 |
| 650 | full, eval seeds | 0, 0, 0, 1, 0 | ≤ 5 s | 0 | 0 |
| **675** | full, eval seeds | 0 on all | 0 s | 0 | 0 |

- There is a **cliff around 700–750 veh/h**: the grid goes from no spillback to self-sustaining gridlock,
  and even a short +20 % peak triggers gridlock that does not recover after the peak.
- 700 veh/h was stable on calibration seeds 1–2 but gridlocked on evaluation seed 10002, so the baseline uses **675 veh/h**.
  That is the highest tested flow meeting all calibration targets on all 5 evaluation seeds:
  < 5 never inserted, no long spillback, 0 teleports.
- The demand was calibrated on the evaluation seeds themselves, because the targets were stated "on all 5 seeds".
  Later controllers must be compared on these same seeds.

## Dropped target: "queues reach the bus stops some of the time"
This target came from the single-junction scenario and is **not used for the grid**. The reasons:
- **Single junction:** the upstream stop is near-side, 150 m before the stop line.
  A normal cycle-scale queue at the junction reaches it, so the target was a sensible congestion check there.
- **Grid:** all stops after the entry stop are **far-side**, 100 m after a junction. A bus stopping there has already crossed the signal.
  - A queue can only reach such a stop by backing up from the *next* junction.
  - That queue must be about 185 m long, about 65 % of a 300 m block, which is far beyond the queue of one cycle.
  - The calibration table shows the stop-queue share rises above 0 only at flows that also spill back and gridlock (≥ 750 veh/h, or a +20 % peak).
  - Meeting the target would therefore have meant abandoning the "not gridlocked" targets.
- **What matters for far-side stops:** the signal affects the bus *before* the stop.
  This is captured by the per-junction bus delay (`seg_delay`, stop `s_k` → `s_{k+1}`) and by its effect on headway CV and bunching downstream.
- The `stopq_reach_*` columns stay in the CSVs as a diagnostic. In the baseline they are 0 on all seeds.

---

# Official results: headless libsumo only (both scenarios)
- **Official results come only from headless runs with libsumo** (SUMO 1.28.0, the pip package in the Python 3.11 venv).
  `--gui` (traci + sumo-gui) is for watching a run, **never for results**.
- Every results CSV has a `backend` column, and it must read `libsumo` on every row.
  - Without `--gui`, the runners fall back to traci when libsumo cannot be imported (wrong interpreter), and that run is then traci.
  - A CSV with `traci` rows is not an official result and must be re-run.
- Why one backend: the traci path drives the SUMO binary in `SUMO_HOME` (1.27.1), and its numbers differ from libsumo 1.28.0 for the same seeds.
  On the grid baseline, seeds 10000–10004, total bunching was 139 / 139 / 115 / 120 / 119 with libsumo vs 136 / 139 / 119 / 123 / 122 with traci.
  Mixing backends would make controller comparisons meaningless.
- Current official baselines, all libsumo on seeds 10000–10004: `results/fixed_time.csv` (single junction) and `results/grid_fixed_time*.csv` (grid).
  The grid baseline was re-run headless and reproduces the libsumo calibration run exactly.

---

# Step 3: rule-based bus priority on the grid (`rule_tsp.py`)
The network, demand and metrics are unchanged. Run with `Grid_Demo.py --controller {fixed,tsp_all,tsp_cond}`.
Results go to `results/grid_<controller>*.csv` (fixed keeps `grid_fixed_time*`), plus `*_decisions.csv` for the TSP runs.

## Structure
- `sense(sim, ...)` is the only code that calls SUMO. It returns a `JunctionView` of plain data.
- `decide(view, params)` is a pure function returning a `Decision` with a reason string. It is unit-tested without SUMO.
- `TspController.act(t, sim)` (one per junction) is the glue. It does the per-cycle extension bookkeeping and the bus-passage
  bookkeeping used for lateness, then applies the decision.

## Detection
- Buses are detected on a junction's incoming approach (2-lane piece + turn-lane piece) within **150 m** of the stop line.
- Excluded: a bus dwelling at a stop, or whose next stop lies ahead on the same edge.
  This only matters for the entry-link stop `s0`, which is about 130 m before the first junction.
- A bus "passes" the junction when it leaves the detection zone, i.e. crosses the stop line (or teleports).

## Actions (both respect the team timing)
- Priority only changes the **remaining time of the current green** (`trafficlight.setPhaseDuration`),
  or ends it immediately by switching to that green's own yellow.
  Yellow (3 s) and all-red (2 s) are never shortened or skipped. The program then continues with its normal durations,
  which is the "return to the normal fixed-time cycle".
- **Green extension:** the bus's phase is green and the estimated time to clear exceeds the remaining green.
  - Time to clear = `dist / max(speed, 5 m/s) + 2 s` (`V_MIN_EST`, `CLEAR_MARGIN`).
  - The extension is the shortfall, rounded up to whole seconds.
  - It is granted only if the whole shortfall fits in the remaining budget (`MAX_EXT` = 15 s per junction per cycle,
    reset when the E-W green starts) **and** the green stays ≤ 60 s.
  - Otherwise it is refused (`budget` / `max_green`). A partial extension the bus cannot use is never given.
- **Early recall:** the conflicting green is running.
  - Its remaining time becomes `max(0, 10 s − spent)`, so a green is never cut below 10 s.
  - Nothing is done if the green would end sooner anyway, or if a yellow/all-red is running.
- Checked in simulation (seed 1, both controllers, all 9 junctions): greens 10–50 s (E-W) and 10–45 s (N-S),
  every yellow exactly 3 s, every all-red exactly 2 s.

## Controller B (`tsp_all`, unconditional)
- Every detected bus that needs it gets an extension or recall. Queues are ignored.
- **Conflicts** (an E-W and an N-S bus at the same time, only possible at `B1`): the bus served by the current green wins,
  since its extension serves it immediately; then the nearest bus.
  This tie rule was not specified in the brief and is my choice.

## Controller C (`tsp_cond`, conditional)
- **Lateness** = `t − (time the previous bus of the same line direction passed this junction) − H` (H = 180 s).
  - Priority only if lateness > `LATE_THR` = 0 s.
  - The first bus of a line direction at a junction has no reference, so it gets no priority.
  - Lateness grows while a bus waits, so a bus that waits long enough becomes "late".
- **Queue refusal:** refused if the **largest** halting queue on any approach that would lose green exceeds `QUEUE_THR` = 15 veh.
  - The losing approaches are the conflicting direction for an extension, and the current green direction for a recall.
  - Interpretation: per approach (max), not summed over approaches.
- **Conflicts:** the later bus wins (tie: higher occupancy).
- All thresholds are constants in `rule_tsp.py` and CLI args (`--detect-dist --max-ext --late-thr --queue-thr`).
  They were **not tuned**: development runs used non-evaluation seed 1, and the evaluation used the defaults.

## Decision log and request counting
- Every decision is written to `results/raw/<tag>/seed_N/decisions.csv` and `results/<tag>_decisions.csv`.
  Columns: t, tls, bus, line, action, code, seconds, lateness, reason. Example reason: `extend 8 s for L1_EB_12, 34 s late`.
  Repeated identical decisions for the same bus are logged once.
- **A request** is one bus pass through one junction during which an action would have been needed.
  Buses that clear in the running green, or arrive during yellow/all-red, make no request.
  - The outcome is "granted" if at least one action was applied during the pass.
  - Otherwise it is "refused", with the last refusal reason. Codes: `not_late`, `queue`, `budget`, `max_green`, `conflict`.
  - Counted for passes in the measurement window.
  - A bus that loses a conflict but is served later in the same pass counts as granted.
    That is why `refused_conflict` is 0 in the evaluation, although the decision logs contain conflict refusals
    (36 for B, 2 for C over 5 seeds).

---

# Step 4: tuning and stress-testing controller C (`c_experiments.py`)
Fixed-time, controller B, the network, the demand and the existing metrics are unchanged.
Re-running fixed-time seed 10000 after the changes reproduces every existing column of `results/grid_fixed_time.csv`.
The one exception is `spill_worst_appr`: with no spillback on any approach (all 0 s), the "worst" approach is an arbitrary tie,
picked in set order.

## Running the experiments
- Every run is a `Grid_Demo.run_seed()` call, headless libsumo (SUMO 1.28.0, the SUMO-RL venv). Every results row is checked for `backend == libsumo`.
- Runs execute in parallel processes, 4 by default (`--workers`). 10 workers ran the machine out of memory.
- Each run's rows are written to the CSVs as soon as it finishes. A restarted experiment skips the runs already in its runs CSV.
- Raw SUMO outputs and route files of these runs are deleted once the metrics are computed. Routes are regenerated from the seeds.

## New metric: cross-street delay
- The "approaches not on a bus line" are the 24 signal approaches whose edges are not on any bus route.
  The other 12 are the bus-line approaches: the middle row and middle column, both directions, including the fringe entries `left1A1`, `right1C1`, `bottom1B0`, `top1B2`.
- **`cross_delay`** = Σ car time loss on those approaches (both pieces, `XY` + `XY.250.00`) / Σ cars that left them (left the turn-lane piece).
  - The source is a car-only SUMO `edgeData` output over the measurement window (600–7800 s).
  - It is the mean delay **per approach passage**, not per car trip: a car crossing several junctions counts once per approach.
  - Time loss counts slowing down, queuing and turning, so it is > 0 even at free flow (about 14 s on the quietest approaches).
- **`bus_appr_delay`** is the same quantity for cars on the 12 bus-line approaches, reported for contrast.
- Insertion backlog (cars that cannot enter) is not in either value. It is reported separately as `never_inserted`.
- `queue_rule_hits` = number of (junction, bus) passes refused at least once by C's queue rule, including passes granted later.
  `refused_queue` counts only passes whose final outcome was a queue refusal.

## Tuning (training seeds 0–9 only)
- The grid is L ∈ {0, 20, 40} s × Q ∈ {8, 12, 15} veh × MAX_EXT ∈ {10, 15, 20} s, 27 settings × 10 seeds, plus fixed-time on the same seeds.
- **"Bunching at the last stop"** = bunched headways (< 0.5 H) at the last stop, summed over the 4 line directions (`bunch_last_total`).
  The objective is its mean over the 10 seeds.
- **"Car delay"** = mean `timeLoss + departDelay` per car (`car_delay_total`), mean over seeds. It must be ≤ 1.03 × the fixed-time mean on the same seeds.
- **Ties** are broken by lower bus junction delay.
- **Result** (`results/c_tuning_summary.csv`). Fixed-time on the training seeds: car delay 75.1 s, last-stop bunching 34.1.

  | rank | setting | last-stop bunching | car delay vs fixed | bus junction delay |
  |---|---|---|---|---|
  | 1 | L 0, Q 12, E 20 | 27.7 | −3.6 % | 22.3 s |
  | 2 | L 0, Q 15, E 15 (= C_default) | 28.0 | −8.1 % | 22.2 s |
  | 3 | **L 0, Q 15, E 20 (chosen)** | 28.2 | −6.5 % | 22.1 s |
  | 4 | L 0, Q 15, E 10 | 29.1 | −8.8 % | 22.4 s |
  | 5 | L 0, Q 8, E 15 | 29.1 | −9.2 % | 22.7 s |

- **Choice: C_tuned = L 0 s, Q 15 veh, MAX_EXT 20 s.** This is not the strict #1, for three reasons.
  - The top 3 are within 0.5 bunched headways, while the standard error of each setting's mean is about 1. They are statistically tied.
  - The strict #1 had a breakdown on training seed 3 (55 cars never inserted). The chosen setting has none on any training seed and the lowest bus delay of the three.
  - The 3 % car-delay constraint is noisy. Fixed-time itself broke down on training seed 0 (50 never inserted), which inflates the reference.
- **What the grid shows:**
  - Every L = 0 setting beats every L = 20 and L = 40 setting. Waiting until a bus is 20–40 s late grants too few requests to reduce bunching.
  - Three L = 40 settings (Q 12 and Q 15 with E 20, and Q 15 with E 15) break down on several seeds.
- `C_default` keeps the step 3 defaults (L 0, Q 15, MAX_EXT 15). The `rule_tsp.py` defaults are unchanged; C_tuned is passed as parameters.

## Stress test (training seeds 0–4)
- Fixed, B and C_tuned at 600, 675 and 725 veh/h per entry. Results in `results/c_stress_summary.csv`.
- **Queue-rule refusals rise with demand:** about 0 / 2.4 / 26.6 refused passes per run at 600 / 675 / 725 veh/h.
- **At 725 veh/h** (beyond the 700–750 calibration cliff), every controller breaks down.
  - C keeps cross-street delay well below B: 43 vs 52 s, with 8 vs 15 teleports.
  - C does not match fixed-time: 36 s cross-street delay, and 401 vs 208 cars never inserted for C vs fixed.
  - So the queue rule protects cross streets relative to unconditional priority, but cannot prevent gridlock at this demand.

## Final evaluation (seeds 10000–10019, 20 seeds)
- Fixed, B, C_default and C_tuned at 675 veh/h. All 80 rows are `backend = libsumo`.
- Results in `results/c_final_summary.csv` (mean and 95 % CI half-width, Student t with df = 19),
  per-seed rows in `c_final_runs.csv`, per stop in `c_final_lines.csv`.
- The CI is across seeds for each controller separately. Controllers share seeds, so the paired differences vs fixed (below) are tighter.

| metric | fixed | B | C_default | C_tuned |
|---|---|---|---|---|
| bus junction delay (s) | 27.7 ± 0.3 | 15.1 ± 0.7 | 22.2 ± 0.3 | 21.9 ± 0.3 |
| bus travel time (s) | 281.1 ± 1.0 | 242.8 ± 2.2 | 264.2 ± 1.1 | 263.1 ± 1.1 |
| last-stop bunching | 35.8 ± 1.9 | 33.6 ± 1.7 | 27.7 ± 1.5 | 27.6 ± 1.9 |
| total bunching | 119.8 ± 6.9 | 101.0 ± 6.1 | 95.7 ± 6.2 | 96.5 ± 6.5 |
| last-stop headway CV | 0.50 ± 0.01 | 0.56 ± 0.01 | 0.43 ± 0.01 | 0.43 ± 0.01 |
| car delay (s) | 72.6 ± 5.6 | 77.2 ± 11.7 | 70.9 ± 3.2 | 68.8 ± 0.7 |
| cross-street delay (s) | 21.8 ± 1.6 | 24.8 ± 3.9 | 21.8 ± 1.0 | 21.2 ± 0.2 |
| bus-line approach car delay (s) | 22.2 ± 1.9 | 19.6 ± 2.2 | 20.0 ± 0.7 | 19.4 ± 0.2 |
| person delay (h) | 561 ± 34 | 524 ± 71 | 524 ± 20 | 510 ± 5 |
| never inserted | 3.1 ± 6.4 | 18.1 ± 36.7 | 0.05 ± 0.10 | 0.05 ± 0.10 |
| teleports | 0.05 ± 0.10 | 0.7 ± 1.5 | 0 | 0 |
| TSP requests / granted | – | 448 / 429 | 530 / 195 | 526 / 197 |
| refused by queue rule | – | 0 | 0.7 | 0.45 |

**Paired difference vs fixed-time** (mean ± 95 % CI over the 20 seeds):

| | last-stop bunching | car delay (s) | cross-street delay (s) | bus junction delay (s) |
|---|---|---|---|---|
| B | −2.2 ± 2.1 | +4.6 ± 13.2 | +3.0 ± 4.3 | −12.7 ± 0.7 |
| C_default | −8.1 ± 1.6 | −1.7 ± 5.1 | −0.1 ± 1.5 | −5.5 ± 0.4 |
| C_tuned | −8.2 ± 1.8 | −3.9 ± 5.5 | −0.6 ± 1.6 | −5.9 ± 0.4 |

- **Breakdowns drive the wide car-delay CIs** for fixed and B: fixed broke down on seed 10009 (61 never inserted), B on seed 10001 (351) and seed 10018 (10).
  Both C variants have 1 never-inserted car, on seed 10010 only, where every controller also has 1.
- **C_tuned and C_default are indistinguishable** on every bus metric. Tuning MAX_EXT 15 → 20 s did not measurably change C.
- **B helps bus speed but not regularity.** It gives the lowest bus delay but raises last-stop headway CV above fixed-time (0.56 vs 0.50),
  because it also speeds up buses that are already early. Its last-stop bunching gain over fixed (−2.2 ± 2.1) is barely significant.
- **C does not shift delay to cross streets** at 675 veh/h. The paired cross-street change is −0.6 ± 1.6 s, while B's is +3.0 ± 4.3 s.
- **Figure:** `results/cv_along_line.png` shows headway CV by stop index (0–3), one panel per line direction, for fixed / B / C_tuned,
  with the mean and 95 % CI over the 20 evaluation seeds.

# Step 5b: the RL agent at all 9 junctions (`rl/evaluate.py --scope all9`)
One trained policy (agent F, observation v2), one independent copy per junction, no
retraining and no communication between junctions. Own libsumo loop as in `--scope b1`
(no SUMO-RL multi-agent mode). `--scope b1` (the default) is unchanged: on seed 900 it
reproduces the stored C_all and F rows on every numeric column.

## Per-junction observation (`rl/sense.py` Geometry, `rl/observation.py` observe_at)
- **Approaches are read from the network** for any junction: each incoming edge that ends at
  the stop line (the 3-lane `.250.00` turn-lane piece) plus its upstream 2-lane main edge.
  The approach id is the main edge (e.g. `left1A1`).
- **Slot order is W, E, S, N for every junction**, from the from-node coordinates relative to
  the junction. For B1 this is exactly the old fixed tuple `A1B1, C1B1, B0B1, B2B1`.
- **Straight exit** (v2 `out_occ`): the outgoing edge reached by the approach's `s` (straight)
  connections in the network, not parsed from edge names (fringe edges such as `left1A1`
  do not follow the grid naming).
- **Missing approach -> zeros**: a side without an approach is a `None` slot and all its
  values (queue, bus, v2 extras) are 0. No junction of this grid has one; all 9 have 4
  approaches, 20 incoming lanes, 8 outgoing lanes, 16 links and the same 6-phase program
  (phase 0 = E-W green, phase 3 = N-S green), so action k means the same green everywhere.
- **B1 is unchanged.** Checked on 360 real decision ticks (seed 900) against the code before
  the change (identical geometry, states, v1 and v2 vectors), and by a pytest on a stored
  60-tick fixture (`rl/tests/data/b1_golden.pkl`).

## What differs between junctions (and the policy never saw at B1)
| | B1 | A1, C1 | B0, B2 | A0, A2, C0, C2 |
|---|---|---|---|---|
| fringe approaches (main edge 246 m, not 234 m) | 0 | 1 | 1 | 2 |
| fringe exits (one 288 m edge, not 234 m) | 0 | 1 | 1 | 2 |
| bus lines | L1 + L2 | L1 only (E-W) | L2 only (N-S) | none |
- Queue features are normalised by each approach's own capacity (fringe: 79.2 cars, others 76.0).
- At the 4 corners every bus feature is always 0: the policy acts on queues and phase only.
- **Bus at the entry stop is visible at A1, C1, B0, B2.** The first stop of each line is on its
  entry link, ~130 m before the line's first junction, inside the 150 m bus range. At B1 the
  nearest upstream stop is ~168 m away, so the policy never saw a bus dwelling within range in
  training. Kept as is (it is what the junction's own sensor sees), not filtered out.
- Headway lateness is measured per junction (last passage of the same line at that junction),
  as at B1; at a line's first junction it reflects the dispatch jitter.

## Each junction's controller
- `RLController(policy, net, tls=J)`: its own Sensor (with its own last-passage memory per
  line), its own B1Signal (shield 10-60 s green, yellow 3 s, all-red 2 s, safety monitor),
  decisions every 5 s after the 600 s warm-up, taking over at its first green.
- Every run row has per-junction `rl_switches_J`, `rl_shield_J`, `rl_violations_J`, and
  per-junction / per-junction-type delays (bus_delay_J where a line crosses J, car_delay_J =
  car time loss on J's approaches / cars that left them, person_delay_J_h).

## Known quirk (pre-existing, not changed)
- `spill_worst_appr` (Grid_Demo) is the arg-max of the longest spillback over approaches;
  `Grid_Demo.approaches()` iterates a Python set, so on ties the reported approach depends on
  the process' string-hash seed. `spill_longest_s` itself is deterministic.
