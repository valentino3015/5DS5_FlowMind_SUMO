# FlowMind: Agent & Coordinator Specifications (Ideas & Objectives)

**Project:** FlowMind (3-Junction Traffic Signal Control in SUMO)  
**Deadlines:** Feature freeze **Saturday 24 Oct 2026** | Final Validation **Wednesday 28 Oct 2026**  
**Core Principles:** Simple beats clever, modular division across 7 teammates, strict contracts in `flowmind/contracts.py`, and zero external framework bloat.

---

## Architecture Overview & Teammate Mapping

FlowMind operates on a 3-junction arterial corridor in SUMO. Every 5 seconds, an orchestrator polls specialist agents, resolves claims via a strict priority hierarchy, routes the candidate action through a rule-based safety shield, and logs decisions with plain-language explanations. Inter-junction coordination is achieved asynchronously through events posted by the corridor coordinator.

| # | Component / Role | Owner             | Primary BO Mapping | Target File(s) | Primary Responsibility |
|---|---|-------------------|---|---|---|
| **1** | **Flow Agent** | Ahmed             | **BO1 & BO2** | `flowmind/agents/flow.py`, `scripts/train.py` | Normal traffic RL (PPO), delay & travel time reduction |
| **2** | **Emergency Agent** | Hadyle            | **BO3** | `flowmind/agents/emergency.py` | Emergency vehicle preemption & clearance |
| **3** | **Bus Agent** | Firas             | **BO4** | `flowmind/agents/bus.py` | Transit signal priority (TSP) & anti-bunching |
| **4** | **Pedestrian Agent** | Omar              | **BO5** | `flowmind/agents/pedestrian.py` | Crossing actuation & maximum wait caps |
| **5** | **Incident Agent** | Donia             | **BO2 & BO7** | `flowmind/agents/incident.py` | Anomaly detection, lane blockage & spillback avoidance |
| **6** | **Safety Shield & Evaluation** | Manoubi           | **BO5, BO6 & BO7** | `flowmind/control/shield.py`, `flowmind/kpis.py` | Safety invariants, HBEFA emissions & audit KPIs |
| **7** | **Corridor Coordinator & Orchestrator** | Adem              | **BO2 & BO7** | `flowmind/control/coordinator.py`, `orchestrator.py` | Multi-junction coordination, events & priority hierarchy |

---

## 1. Flow Agent (Normal Traffic RL)
**Owner:** Ahmed  
**Target Code:** `flowmind/agents/flow.py`, `scripts/train.py`  
**Objectives:** **BO1** (Reduce Average Vehicle Delay & Improve Travel-Time Predictability) & **BO2** (Corridor Throughput)

### Core Ideas (9)
1. **PPO Reinforcement Learning:** Train a discrete-action PPO model via `stable-baselines3` for green phase selection every 5 seconds.
2. **Normalized Observation Space:** Vectorize incoming lane queue counts, total waiting times, and vehicle occupancies normalized by lane capacity, concatenated with current phase one-hot encoding.
3. **Low-Urgency Baseline Claim:** Always return a background claim `Claim(urgency=0.1, reason="Normal traffic flow optimization")` so it drives the intersection whenever specialist agents are idle.
4. **Action Masking Fallback:** Check `allowed[i]` from the shield before applying the PPO output; if the selected phase is prohibited, fall back to `self.keep_phase(state, allowed)`.
5. **Reward Function Engineering:** Implement a multi-term reward combining negative waiting time delta and negative queue length: $R = -(\sum \Delta W_{lane} + \alpha \sum Q_{lane})$.
6. **Multi-Seed Training & Evaluation:** Train and benchmark across 10 fixed random seeds (`[100..109]`) to ensure statistical reliability and policy stability.
7. **Demand-Pattern Generalization:** Train against varied demand profiles: light traffic, symmetric rush hour, and asymmetric peak commute (heavy inbound morning, heavy outbound evening).
8. **Delay Distribution Reporting:** Measure and report travel-time mean, 95th-percentile, and variance (predictability metric) rather than just average delay.
9. **Fixed-Time Baseline Comparison:** Benchmark directly against the fixed-time controller across identical vehicle injection sequences.

### Backup Ideas (5)
1. **Rule-Based Max-Pressure Controller:** Pure analytical max-pressure heuristic ($P = Q_{in} - Q_{out}$) as a zero-shot, parameter-free fallback if RL training fails to converge before Oct 24.
2. **Deep Q-Network (DQN) Alternative:** Replace PPO with discrete DQN with an experience replay buffer if policy gradient updates exhibit high variance on discrete phase selections.
3. **Queue Moving Average Smoothing:** Apply a 3-step exponential moving average over lane queues to eliminate high-frequency noise from vehicle stop-and-go behavior.
4. **Actuated Green Extension Heuristic:** Simple gap-based extension (extend green by 5s if vehicles are detected within 30m of the stop line, up to `max_green`).
5. **Short-Term Demand Forecasting Feature:** Compute lane queue growth rates ($dQ/dt$) and append to the observation vector to give the agent anticipatory awareness.

### Rejected Ideas
- **Multi-Agent Deep RL with Communication Channels:** Directly violates `CLAUDE.md` Rule 2 ("Never use SUMO-RL multi-agent in flowmind").
- **Individual Vehicle Dynamic Rerouting:** Outside signal control scope; traffic lights control phases, not vehicle navigation.
- **LLM-Generated Phase Decisions:** Directly violates `CLAUDE.md` Rule 3 ("No LLM calls anywhere in flowmind/").
- **Variable Speed Limits (VSL):** Requires additional roadside infrastructure beyond signal control.

---

## 2. Emergency Agent (Priority & Preemption)
**Owner:** Hadyle  
**Target Code:** `flowmind/agents/emergency.py`  
**Objectives:** **BO3** (Reduce Emergency Response Time While Maintaining Safety)

### Core Ideas (9)
1. **Special Vehicle Proximity Scan:** Inspect `state.special_vehicles` for entries with `kind == "emergency"` on each 5-second tick.
2. **Maximum Urgency Claim:** When an emergency vehicle is detected within `detect_distance` (default 200m), return `Claim(urgency=1.0, reason=f"Emergency vehicle {veh_id} within {dist:.0f}m on {lane_id}")`.
3. **Approach-Phase Mapping:** Map the emergency vehicle's incoming lane to its corresponding green phase using `self.phase_for_lane(state, veh.lane_id, allowed)`.
4. **Immediate Phase Call:** Command the designated emergency green phase immediately if allowed by the safety shield.
5. **Phase Hold Until Stop-Line Passage:** Maintain the emergency green until the vehicle's stop-line distance drops below 5m or it exits the incoming lane.
6. **Safety Clearance Invariance:** Rely on `env.py` and `shield.py` to strictly enforce yellow (3s) and all-red (2s) intervals before switching, preventing physical collisions.
7. **Coordinator Event Integration:** Process incoming `Event(kind="emergency")` posted by the corridor coordinator to prepare the junction before the vehicle enters detection range.
8. **Stuck-Request Protection:** Expire emergency priority if the vehicle remains stationary or un-cleared after a maximum timeout (e.g. 90s) to prevent permanent lockup.
9. **Cost-of-Priority KPI Tracking:** Calculate secondary traffic delay incurred on non-priority approaches during the preemption window.

### Backup Ideas (5)
1. **Upstream Queue Flushing:** If a coordinator alert gives an ETA of 15-20s, activate the emergency phase 10s early to flush standing queues ahead of the ambulance.
2. **Simultaneous Emergency Conflict Resolution:** If two emergency vehicles approach on conflicting approaches, prioritize the one with the shortest estimated time to arrival ($dist / speed$).
3. **Graceful Post-Preemption Phase Handoff:** After emergency passage, immediately hand green to the cross approach that accumulated the highest waiting time.
4. **Speed-Adaptive Detection Window:** Dynamically adjust the detection distance threshold based on emergency vehicle approach speed ($dist_{trigger} = \max(150, speed \times 12)$).
5. **Pedestrian Interruption Mitigation:** Delay emergency phase initiation by up to 3 seconds if pedestrians are already halfway across a conflicting crosswalk.

### Rejected Ideas
- **Zero-Second Clearance (Instant Green):** Cutting yellow/all-red intervals causes physical vehicle collisions in SUMO.
- **Autonomous Vehicle Lane-Clearing:** Requires V2V connected vehicle penetration not available in standard SUMO driver models.
- **Real EMS Dispatch Integration:** Out of scope; simulated vehicle arrivals are used.
- **Drone-Assisted Aerial Routing:** Outside the domain of intersection signal control.

---

## 3. Bus Agent (Transit Priority & Anti-Bunching)
**Owner:** Firas  
**Target Code:** `flowmind/agents/bus.py`  
**Objectives:** **BO4** (Improve Public Transport Reliability & Reduce Bus Bunching)

### Core Ideas (9)
1. **Conditional Transit Signal Priority (TSP):** Scan `state.special_vehicles` for `kind == "bus"` and only request priority if `delay >= min_delay` (e.g. 60s behind schedule).
2. **Proximity Filtering:** Only trigger priority if the bus is within `detect_distance` (e.g. 150m) of the junction stop line.
3. **Dynamic Urgency Scaling:** Scale urgency proportionally to lateness: `urgency = min(1.0, delay / 300.0)`, ensuring heavily delayed buses outrank mildly delayed ones.
4. **Dual TSP Strategy (Extension vs Early Green):** If the current phase already serves the bus, extend green (`keep_phase`); if red, request early phase recall via `phase_for_lane()`.
5. **Cross-Street Congestion Veto:** Suppress or scale down priority claim if cross-street queue exceeds a saturation threshold (e.g. queue > 10 vehicles) to prevent corridor gridlock.
6. **Person-Delay Optimization:** Weight priority claims by estimated bus passenger occupancy (e.g. 30 passengers per bus vs 1.2 per private car).
7. **Bus Stop Dwell-Time Awareness:** Invalidate or delay priority request if the bus is currently decelerating into or stopped at a scheduled bus stop.
8. **Headway Regularity Monitoring:** Compute elapsed headway between successive buses along the corridor to detect bunching early.
9. **Comparative KPI Evaluation:** Directly benchmark total person-delay vs vehicle-delay outcomes against the fixed-time baseline.

### Backup Ideas (5)
1. **Anti-Bunching Priority Denial:** If a bus is running ahead of schedule or following too closely (< 120s) behind a leading bus, actively deny priority to restore regular spacing.
2. **Maximum Priority Green Cap:** Enforce a strict ceiling (e.g., maximum 15s of additional green extension) to bound disruption to cross traffic.
3. **Multi-Bus Queue Resolution:** If multiple buses arrive on different approaches, serve the approach with the highest combined passenger delay ($\sum delay \times passengers$).
4. **Coordinator Bus Progression:** When a late bus clears J1, coordinator posts an event to J2 with estimated travel time so J2 can synchronize green.
5. **Bus Lane Simulation Benchmark:** Evaluate performance on dedicated bus-lane scenarios versus mixed-traffic scenarios to quantify infrastructure synergy.

### Rejected Ideas
- **Unconditional Priority for All Buses:** Causes excessive private vehicle delay and network collapse when buses are on-schedule.
- **Timetable and Route Redesign:** Operator-level logistics, outside the scope of traffic signal software.
- **Live Transit Operator AVL Feed Integration:** Out of scope; delay is extracted directly from simulation contracts.
- **Holding Buses at Green Lights to Fix Bunching:** Holding must occur at stops, not by arbitrarily holding green/red on open roads.

---

## 4. Pedestrian Agent (Crossing Safety & Waiting Caps)
**Owner:** Omar  
**Target Code:** `flowmind/agents/pedestrian.py`, `flowmind/control/shield.py`  
**Objectives:** **BO5** (Improve Pedestrian Safety & Reduce Unnecessary Pedestrian Waiting)

### Core Ideas (9)
1. **Crossing Actuation & State Inspection:** Inspect `state.crossings` every 5s for waiting pedestrians (`crossing.waiting >= min_waiting`, e.g. >= 1 person).
2. **Waiting-Time Threshold Claim:** Trigger a claim if `crossing.max_wait >= max_wait` (default 60s) with `urgency = 0.65`.
3. **Anti-Starvation Escalation:** Dynamically escalate urgency as wait increases: `urgency = min(0.95, 0.65 + 0.3 * (max_wait - 60) / 30)`, reaching near-emergency urgency at 90s.
4. **Phase-Crossing Correlation:** Map the target crossing to a compatible green phase using `self.phase_for_crossing(state, crossing.crossing_id, allowed)`.
5. **Demand-Responsive Only:** Return `None` if no pedestrians are waiting, avoiding unnecessary vehicle interruptions.
6. **Multi-Crossing Synchronization:** If pedestrians are waiting at multiple crossings, select the phase that safely clears the greatest number of waiting pedestrians simultaneously.
7. **Clearance Speed Assumption:** Assume reduced-mobility walking speed ($0.8 \text{ m/s}$) to guarantee adequate clearance time before phase changes.
8. **Waiting Distribution Metrics:** Report maximum wait and 95th-percentile wait in addition to average wait to highlight worst-case equity.
9. **Deliberate Baseline Failure Experiment:** Train and evaluate an agent without pedestrian awareness to empirically demonstrate safety risks and starvation under unconstrained flow optimization.

### Backup Ideas (5)
1. **Crowd-Size Urgency Multiplier:** Boost claim urgency when large groups assemble: `urgency += min(0.2, 0.04 * crossing.waiting)`.
2. **Pedestrian Red-Light Compliance Proxy:** Track instances where waiting time exceeds 100s as a proxy for unsafe jaywalking / red-light crossing risk.
3. **Pedestrian Call Extension:** If newly arriving pedestrians are detected while crossing green is active and clearance allows, extend green by up to 5s.
4. **School / Rush-Hour Schedule Mode:** Apply a time-of-day configuration reducing the `max_wait` threshold from 60s to 40s during designated school-zone periods.
5. **Push-Button Request Acknowledgment Logging:** Log instantaneous request timestamp in `logger.py` to audit response latency between button push and green onset.

### Rejected Ideas
- **Smartphone Bluetooth / Wi-Fi Pedestrian Tracking:** Severe privacy barriers, adoption friction, and out of project scope.
- **Physical Crosswalk Infrastructure Reconstruction:** Infrastructure modification rather than intelligent signal control.
- **Audible Signal Accessibility Hardware Simulation:** Hardware deployment outside software scope.
- **Indefinite Pedestrian Green Extension:** Starves vehicle traffic completely.

---

## 5. Incident Agent (Anomaly Detection & Spillback)
**Owner:** Donia  
**Target Code:** `flowmind/agents/incident.py`  
**Objectives:** **BO2** (Throughput & Gridlock Prevention) & **BO7** (Anomaly Detection & Auditable Handling)

### Core Ideas (9)
1. **Zero-Flow Anomaly Detection:** Flag an incident if a lane has `queue >= min_queue` (default 5 vehicles), `occupancy > 0.8`, and zero vehicle departures for `halt_time >= 60s`.
2. **High-Urgency Anomaly Claim:** Return `Claim(urgency=0.85, reason=f"Incident suspected on {lane_id}: halted for {time:.0f}s with queue {queue}")`.
3. **Green Phase Diversion:** Identify phases in `state.phase_lanes` that feed the blocked lane and command alternative phases that serve unblocked, conflict-free approaches.
4. **Upstream Spillback Mitigation:** Coordinate with the corridor coordinator to alert the upstream junction to throttle green into the blocked segment.
5. **Box-Blocking / Downstream Exit Check:** Inspect `state.out_lanes`; if an outgoing lane is 100% occupied, suppress incoming green phases that discharge into it.
6. **Automatic De-escalation & Release:** Withdraw claim immediately once vehicles resume moving on the lane (average lane speed > 2.0 m/s for two consecutive cycles).
7. **Coordinated Emergency Passage Support:** When an incident occurs simultaneously with an emergency claim, defer immediately to the emergency agent.
8. **Synthetic Incident Evaluation Scenario:** Create a reproducible SUMO test scenario injecting a 120-second artificial lane stoppage (`traci.vehicle.setStop`) to benchmark incident response.
9. **Gridlock Prevention Metric:** Record the frequency and duration of gridlock events (where all junction movements become zero-speed).

### Backup Ideas (5)
1. **Adjacent-Lane Occupancy Discrepancy Heuristic:** Detect an incident by comparing lane occupancies on the same approach (e.g. Lane 0 at 95% occupancy while Lane 1 is at 15%).
2. **Proactive Queue Metering:** Progressively shorten green times on upstream feeding lanes rather than an abrupt shutoff, preventing sudden shockwaves.
3. **Multi-Junction Bottleneck Ranking:** Compute a real-time bottleneck index across J1, J2, and J3, dynamically logging the critical restriction point in `logger.py`.
4. **Rolling Average Speed Threshold:** Trigger an incident claim if mean lane speed drops below 0.5 m/s despite continuous green allocation.
5. **Incident Recovery Green Wave:** Once the obstacle clears, trigger a temporary extended green phase to rapidly flush the residual backlog.

### Rejected Ideas
- **Computer Vision on Raw Video Camera Feeds:** SUMO provides exact lane telemetry; raw pixel processing adds unneeded complexity.
- **Dynamic Vehicle In-Car Rerouting:** Outside signal controller capabilities.
- **Deep Learning Autoencoders for Anomaly Detection:** Violates the "Simple beats clever" guideline; deterministic rule thresholds are robust and debuggable.
- **Variable Message Signs (VMS) Driver Warnings:** Requires roadside display infrastructure simulation.

---

## 6. Safety Shield & Evaluation / Emissions Specialist
**Owner:** Manoubi  
**Target Code:** `flowmind/control/shield.py`, `flowmind/kpis.py`, `scripts/evaluate.py`  
**Objectives:** **BO5** (Safety Invariants), **BO6** (Emissions & Fuel), **BO7** (Auditable & Trustworthy Guard)

### Core Ideas (9)
1. **Physical Safety Invariants Enforcement:** Strictly enforce `min_green` (10s) and `max_green` (60s) for all phases regardless of which agent is commanding.
2. **Dynamic Legal Phase Filtering:** Pre-compute the boolean `allowed: list[bool]` array passed to `Specialist.act()`, preventing agents from requesting illegal phases.
3. **Active Pedestrian Clearance Guard:** Monitor `Crossing.on_crossing`; block phase transitions while pedestrians are actively walking, verifying clearance time via `ped_clearance_speed = 0.8 m/s`.
4. **Action Overwrite & Audit Logging:** If an agent returns an unallowed phase index, override it with a safe phase and log the exact invariant violation in `logger.py`.
5. **SUMO HBEFA Emission Modeling:** Integrate SUMO’s HBEFA emission models to log total CO2, NOx, PM, and fuel consumption per vehicle and scenario.
6. **Multi-Objective Pareto Frontier Analysis:** Generate Pareto plots mapping average vehicle delay against total CO2 emissions across controllers.
7. **Comprehensive Audit KPIs:** Compute mean, 95th-percentile, and variance for waiting times, queue lengths, bus schedule deviation, and ambulance travel times.
8. **Deterministic Fixed-Time Fallback:** If any agent raises an unhandled exception or returns an invalid action, the shield automatically switches the junction to a failsafe fixed-time cycle.
9. **Zero Safety Violations Assertion:** Run an automated verification script asserting exactly zero red-light conflicts and clearance violations across all test runs.

### Backup Ideas (5)
1. **Harsh-Acceleration & Stop-Count Proxies:** If HBEFA calculation adds overhead, compute stop frequency and harsh accelerations (> 2.5 m/s²) as lightweight emission proxies.
2. **Modal Emission Breakdown:** Report emissions separately for private passenger cars, diesel transit buses, and emergency vehicles.
3. **Emission Hotspot Corridor Mapping:** Aggregate emissions per 50-meter corridor segment to identify intersection queue emission hotspots.
4. **Out-of-Distribution (OOD) Operational Alert:** Trigger a warning log entry if incoming demand exceeds 3 standard deviations from training distribution.
5. **Automated Markdown Executive Summary:** Automatically compile evaluation results, tables, and comparison percentages into an exportable summary report.

### Rejected Ideas
- **Atmospheric Air Dispersion Fluid Dynamics Modeling:** Belongs to environmental science; SUMO tailpipe emissions are sufficient.
- **Formal Mathematical Theorem Proving of Neural Policies:** Infeasible within student timeframe; run-time safety shielding is the industry standard.
- **Overriding Clearance Intervals During Emergencies:** Bypassing yellow/all-red causes physical collisions in SUMO.
- **Regulatory ISO/IEC Certification Compliance Workflows:** Beyond student deliverable scope.

---

## 7. Corridor Coordinator & Orchestrator
**Owner:** Adem  
**Target Code:** `flowmind/control/orchestrator.py`, `flowmind/control/coordinator.py`, `flowmind/logger.py`  
**Objectives:** **BO2** (Corridor Throughput & Green Waves) & **BO7** (Auditable Orchestration & System Trust)

### Core Ideas (9)
1. **Deterministic Priority Arbitration:** Every 5 seconds, poll all specialist agents (`wants_control`) and grant control via strict hierarchy: `emergency > incident > pedestrian > bus > flow`.
2. **Decision Hysteresis (Anti-Flicker):** Enforce `min_hold = 10s` to prevent rapid high-frequency switching between competing specialist agents.
3. **Inter-Junction Event Dispatching:** Monitor vehicle passages at J1, J2, and J3, creating and transmitting `Event` objects across junctions with expiration timestamps (`expires_at`).
4. **Ambulance Corridor Progression:** When an emergency vehicle is detected approaching J1 heading towards J2, compute ETA ($dist / speed$) and post an `Event(kind="emergency", target_tls="J2")`.
5. **Corridor Spillback Prevention Event:** When a downstream link between J1 and J2 exceeds 85% occupancy, post an `Event(kind="incident")` to J1 to throttle incoming arterial green.
6. **Green-Wave Offset Synchronization:** Synchronize arterial phase progression along J1 -> J2 -> J3 based on free-flow travel time between intersections.
7. **Explainable Plain-Language Logging:** Log every single decision with structured human-readable justification: `[Step 140 | J2] Active: emergency | Phase: 0 | Reason: amb_1 approaching in 18s | Shield: Approved`.
8. **Reproducible Scenario Replay:** Ensure simulations can be re-run with identical random seeds to verify agent decisions under identical vehicle injection.
9. **Manual Operator Override Hook:** Provide an interface function allowing an external operator or test harness to manually command a specific phase.

### Backup Ideas (5)
1. **Static Time-Distance Progression Chart:** Calculate fixed arterial offsets from road lengths and speed limits as a baseline progression plan.
2. **Cross-Junction Late Bus Coordination:** When a bus is > 90s late, notify downstream junctions to advance the transit green phase window.
3. **Multi-Emergency Collision Avoidance:** If two emergency vehicles approach different junctions in the corridor, resolve priority based on route direction and destination.
4. **Decision Latency Monitoring:** Measure execution time per decision step (asserting < 50ms per tick) to ensure real-time viability.
5. **What-If Baseline Comparison:** For every orchestrated step, log what the fixed-time controller would have done in parallel to quantify decision-level delta.

### Rejected Ideas
- **Centralized Deep Multi-Agent RL Controller:** Directly violates `CLAUDE.md` Rule 2 and exceeds timeframe.
- **LLM-Powered Agent Negotiation / Chat:** Directly violates `CLAUDE.md` Rule 3 ("No LLM calls").
- **City-Wide Network Control:** Restricted to the 3-junction corridor scope.
- **Automated Dynamic Vehicle Rerouting:** Outside traffic signal controller domain.
