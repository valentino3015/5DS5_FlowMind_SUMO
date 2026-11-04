# FlowMind

Multi-agent traffic-light control in SUMO. Instead of fixed timers, specialist **agents** decide
when each light changes; a **manager** picks one per junction, a **safety shield** checks every
decision and a **coordinator** warns the next junctions. Every decision is logged and can be
explained (XAI layer, optional local LLM, Gemini assistant).

Team integration branch: **`Adem`**.

## 1. Install on a new PC (Windows)

**Needs:** Python **3.12.1 or newer** (3.12.0 has a zipfile bug that stops the trained `.zip`
models from loading) and Git. SUMO itself comes from pip (`eclipse-sumo`).

```powershell
git config --global core.longpaths true      # before cloning: sumo_rl/nets has very long file names
git clone -b Adem https://github.com/valentino3015/5DS5_FlowMind_SUMO.git
cd 5DS5_FlowMind_SUMO
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt     # ~1 GB (PyTorch)
.venv\Scripts\python.exe -m pytest -q tests                      # all tests must pass
```

Use `.venv\Scripts\python.exe` directly: no need to activate the venv (PowerShell may block it).

Optional:
- **Gemini assistant:** `copy .env.example .env`, then put your own key in `GEMINI_API_KEY`
  (`.env` is never sent to GitHub). Without it everything works except the assistant page.
- **Local LLM for the XAI panel:** install [Ollama](https://ollama.com), then `ollama pull qwen2.5:3b`
  (2 GB). Without it the XAI panel gives its rule-based answer.

## 2. Run it

**Dashboard** (backend and frontend in one program, no Node.js, no database):

```powershell
.venv\Scripts\python.exe scripts\dashboard.py            # http://127.0.0.1:8501 (other port: --port 8600)
```

From the page: run any method on any scenario, watch the agents live, optionally compare with
the fixed timer and max-pressure (ticked baselines run first, without the SUMO window), read the XAI story of a run, ask questions about it. `Lancer-dashboard.cmd` starts the
same thing with the `python` on the PATH.

**One simulation from the command line:**

```powershell
.venv\Scripts\python.exe scripts\run.py --controller orchestrated --scenario corridor_demo --gui
.venv\Scripts\python.exe scripts\run.py --controller fixed_time --scenario corridor --seed 100 --seconds 1800
```

Controllers: `orchestrated` (the full FlowMind system), `fixed_time`, `green_wave`,
`max_pressure`, `flow`, `rules`. Results go to `runs/<date>_.../` (`decisions.csv`,
`decisions.log`, `summary.csv`).

**Explain a run (XAI):**

```powershell
.venv\Scripts\python.exe scripts\explain.py RUN_DIR --story
.venv\Scripts\python.exe scripts\explain.py RUN_DIR --at J2 160
.venv\Scripts\python.exe scripts\explain.py RUN_DIR --ask "why did J2 switch at 160?" --llm
```

**Compare every controller over the evaluation seeds:**

```powershell
.venv\Scripts\python.exe scripts\corridor_report.py --workers 2
.venv\Scripts\python.exe scripts\corridor_report.py --scenario corridor5 --workers 2
```

Keep `--workers` at 2 on a laptop: each worker is one SUMO simulation.

## 3. Scenarios

| Name | What it is |
|---|---|
| `corridor` | 3 junctions J1-J3, 30 min: random traffic, a bus line both ways, pedestrians, ambulance `amb_1` at 120 s, accident between J1 and J2 from ~700 s for 5 min, `amb_2` at 1100 s. **Scenario of the reported results: do not change it.** |
| `corridor_demo` | Same street, 10 min, every agent acts: ambulance 60 s, pedestrian rush at J2 140 s, accident 230 s, late + bunched buses 330 s, 2nd ambulance 430 s. |
| `corridor5`, `corridor5_demo` | Same design with 5 junctions J1-J5 (accident between J2 and J3, pedestrian rush at J3), and **bus stop zones**: buses stop just after each junction for passengers, at least 10 s and 2.5 s per person boarding or alighting, so a late bus finds more people and gets later (bunching). |
| `grid10`, `grid10_demo` | 10 junctions: two parallel main streets J1-J5 (top) and J6-J10 (bottom), 200 m apart, joined by the side streets. Buses with stop zones on both streets. The ambulance turns from the top street down to the bottom one at J3, so its green corridor follows it round the corner. Accident on the top street between J2 and J3. |
| `3x3grid`, `cologne3` | Cars only (from SUMO-RL): flow-agent training and the emission studies. |

Details and how to rebuild a map: [scenarios/README.md](scenarios/README.md).

## 4. Results (mean of seeds 100-102)

**Read this first.** The fixed timer is a weak reference: it gives every phase the same 30 s
(the main road gets half of the green while it carries about 3x more cars) and its junctions are
not coordinated. Real adaptive systems are usually reported to gain about 10-25 % over a well-tuned plan. The fair
reference for cars is **max-pressure**, a strong adaptive method from the literature (it ignores
ambulances, buses and pedestrians):

| Demo (10 min) | Cars waiting | Ambulances | Pedestrian waiting | Bus waiting |
|---|---|---|---|---|
| `corridor_demo` | 32.6 vs 34.9 s (**-7 %**) | 60-63 vs 61-89 s | 18.8 vs 108 s (**-83 %**) | 13.1 vs 16.2 s |
| `corridor5_demo` | 25.0 vs 26.3 s (**-5 %**) | 91-111 vs 115-144 s | 15.4 vs 64 s (**-76 %**) | 20.9 vs 22.7 s |
| `grid10_demo` | 29.0 vs 30.5 s (**-5 %**) | `amb_1` 136 vs 175 s (**-22 %**); max-pressure's `amb_2` never arrives in 10 min | 11.4 vs 32 s (**-65 %**) | 23.3 vs 22.9 s (same) |

For cars FlowMind is a little better than max-pressure (its flow agent is pressure-based too); its
real gain is what the specialists add: ambulances through faster and more reliably, and pedestrians
served (max-pressure leaves them waiting up to minutes).

FlowMind vs the fixed timer:

| Scenario | Cars waiting | Ambulance trips | Bus waiting | Pedestrian waiting |
|---|---|---|---|---|
| `corridor` (30 min) | 20.9 vs 35.1 s (**-40 %**) | 61 vs 81 s (**-25 %**) | 29.3 vs 46.8 s (**-37 %**) | 5.6 vs 24.7 s (**-77 %**) |
| `corridor_demo` (10 min) | 32.6 vs 30.4 s (+7 %) | `amb_1` 60 vs 124 s (**2x faster**) | 13.1 vs 39.9 s (**-67 %**) | 18.8 vs 49.5 s (**-62 %**) |
| `corridor5` (30 min) | 23.0 vs 43.4 s (**-47 %**) | 100-105 vs 179-181 s (**-42 to -44 %**) | 35.5 vs 128.3 s (**-72 %**) | 5.6 vs 24.3 s (**-77 %**) |
| `corridor5_demo` (10 min) | 25.0 vs 34.3 s (**-27 %**) | 91-110 vs 162-179 s (**-32 to -49 %**) | 20.9 vs 104.7 s (**-80 %**) | 15.4 vs 46.4 s (**-67 %**) |
| `grid10` (30 min) | 30.5 vs 61.4 s (**-50 %**) | 115-137 vs 196-206 s (**-33 to -41 %**) | 33.9 vs 137.0 s (**-75 %**) | 4.4 vs 23.1 s (**-81 %**) |
| `grid10_demo` (10 min) | 29.0 vs 50.9 s (**-43 %**) | `amb_1` 136 vs 256 s (**-47 %**) | 23.3 vs 110.7 s (**-79 %**) | 11.4 vs 44.2 s (**-74 %**) |

Bus waiting = time a bus stands still in traffic (SUMO does not count its stops for passengers).
Passengers on `corridor5` wait about the same at the stop with both controllers (240 vs 244 s in
30 min): the buses leave the terminus at random, and signal priority cannot fix that.

Cars in the 3-junction demo: while the accident blocks the road, every adaptive controller loses
to the fixed timer there. On 5 and 10 junctions the fixed timer gridlocks (200-250 vehicles teleported
in 30 min on 5 junctions, 430-470 on the grid, against 7-14 with FlowMind), which also makes its waiting figures look better than they are.

Emission controller (Manoubi): see [docs/ENTRAINEMENT_EMISSIONS_FR.md](docs/ENTRAINEMENT_EMISSIONS_FR.md);
the active model `runs/models/emission_ppo/` is in Git (the only exception to `runs/`).

## 5. Who does what

| Person | Part | Files |
|---|---|---|
| Adem | Orchestrator, corridor coordinator, decision logger, runner, corridor scenarios, XAI layer, team part of the dashboard, integration | `flowmind/control/orchestrator.py`, `coordinator.py`, `flowmind/logger.py`, `scripts/run.py`, `scenarios/corridor*/`, `xai/`, `scripts/explain.py` |
| Ahmed | SUMO environment, flow agent (PPO on the 3x3 grid, heuristic V4 on the corridors) | `flowmind/env.py`, `flowmind/agents/flow.py`, `scripts/train.py` |
| Hadyle | Emergency agent | `flowmind/agents/emergency.py` |
| Firas | Bus agent (BO4 conditional priority) | `flowmind/agents/bus.py`, `sandbox/bo4_bus/` |
| Omar | Pedestrian agent and safety shield | `flowmind/agents/pedestrian.py`, `flowmind/control/shield.py` |
| Donia | Incident agent (detection + PPO response, rule fallback) | `flowmind/agents/incident.py`, `scripts/train_incident.py` |
| Manoubi | Emission controller, evaluation, dashboard, Gemini assistant | `flowmind/evaluation.py`, `flowmind/emission_rl.py`, `dashboard/`, `chatbot/` |

## 6. How a decision happens (every 5 s, at each junction)

1. `env.py` reads SUMO and builds a `JunctionState` (`flowmind/contracts.py`).
2. The coordinator adds warnings: ambulance coming (with its lane and ETA), incident or spillback next door
   (neighbours = junctions joined by a road, so it works on the corridors and on the grid).
3. Every agent says if it wants control (`wants_control`).
4. The orchestrator picks one by priority **emergency > incident > pedestrian > bus > flow**
   (urgency breaks ties; a new agent keeps control at least `min_hold`). Nobody: max-pressure.
5. That agent chooses the green phase (`act`) and says why (`explain`).
6. The safety shield checks it (min/max green, pedestrians on a crossing, transitions).
7. `env.py` inserts yellow and all-red; the logger saves the decision, the reason, who else asked,
   and what a fixed timer would have done.

## 7. Writing or changing an agent

An agent is one class with 3 functions (`flowmind/agents/base.py`): `wants_control`, `act`,
`explain`. Copy the style of `flowmind/agents/baselines.py`. All numbers go in your section of
`config.yaml`, never in the code. Check it with `scripts\run.py --gui` and compare with
`fixed_time`. Don't change `contracts.py` without telling everyone.

## 8. Folders

```
config.yaml        all settings (times, thresholds), one section per agent
flowmind/          contracts, env (the only SUMO connection), agents/, control/, logger, KPIs, evaluation
scenarios/         maps and traffic
scripts/           run, dashboard, explain, reports, training
xai/               explanations from the decision log (+ optional local LLM)
chatbot/           Gemini assistant
dashboard/         the web page
tests/             pytest
sandbox/           prototypes (Ahmed's first demos, Firas's BO4 study)
runs/              results, not on GitHub (except runs/models/)
```

Rules: work in your own files, use your own branch, never upload `runs/` (except
`runs/models/`) or big files.

## Documentation

[COMMANDS.md](COMMANDS.md) (command cheat sheet), [EVALUATION_FR.md](EVALUATION_FR.md),
[docs/CHATBOT_FR.md](docs/CHATBOT_FR.md) (Gemini assistant), [IDEAS.md](IDEAS.md) (design ideas per agent).
