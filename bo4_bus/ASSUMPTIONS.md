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
- **Evaluation seeds are 10000�10004** (team rule: evaluation seeds start at 10000). Seeds below 10000
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
Calibration was done with the earlier 71 s cycle (no all-red) on seeds 1�5. The demand was **not** re-tuned
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
