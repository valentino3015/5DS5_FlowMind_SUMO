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
- Warm-up runs from 0 to 600 s; measurement runs from 600 to 7800 s. libsumo is used when headless and traci with `--gui`.
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

## Baseline run provenance
- `results/grid_fixed_time*.csv` was produced with the **traci backend**, i.e. the SUMO 1.27.1 binary in `SUMO_HOME`.
  The runner falls back to traci when libsumo cannot be imported.
- The venv's libsumo is SUMO 1.28.0. Its numbers for the same seeds differ slightly; for example, total bunching on seeds 10000–10004 was
  139 / 139 / 115 / 120 / 119 with libsumo vs 136 / 139 / 119 / 123 / 122 with traci.
- Controllers must be compared on the same backend and SUMO version as this baseline. The `backend` column records which one was used.
