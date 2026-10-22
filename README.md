# FlowMind: Team Guide

## Modèle RL dédié aux émissions

Entraînement PPO, fine-tuning et comparaison des trois méthodes : [guide français](docs/ENTRAINEMENT_EMISSIONS_FR.md).
Le modèle de vérification fourni est expérimental, sans preuve de gain de performance.

## Dashboard local et évaluation

Double-cliquez sur `Lancer-dashboard.cmd` ou lancez `python scripts/dashboard.py`.
Le navigateur permet de lancer SUMO, enregistrer les métriques et comparer les
contrôleurs disponibles. Instructions complètes : [EVALUATION_FR.md](EVALUATION_FR.md).

Pour une campagne sans interface :

```powershell
python scripts/evaluate.py --seconds 600 --seeds 100 101 --controllers fixed_time max_pressure flow_heuristic
```

**Validation: Wed 28 Oct. Everything must work by Sat 24 Oct.**

## 1. The project in one minute

We simulate a street with **3 traffic lights** in SUMO. Instead of fixed timers, **agents** decide when each light changes.

Think of each junction as a small team:

- **The manager (orchestrator)** looks at the situation and picks who is in charge right now.
- **The experts (agents)** each handle one situation:
  - **Flow:** normal traffic. This one learns by itself (reinforcement learning).
  - **Emergency:** an ambulance is coming.
  - **Bus:** a bus is late.
  - **Pedestrian:** people are waiting to cross.
  - **Incident:** an accident is blocking a lane.
- **The safety guard (shield)** checks every decision and blocks anything dangerous.
- **The coordinator** warns the next junctions, e.g. "an ambulance will reach you in 18 seconds".

At the end, we compare our system with normal fixed-timer lights and show it's better.

## 2. Who does what

| Person | Task |
|---|---|
| Adem | Orchestrator (manager) + corridor coordinator, and the SUMO connection everyone uses |
| TBD | Flow agent (learning) |
| TBD | Evaluation, emissions + insights (results, KPIs, charts) |
| TBD | Emergency agent |
| TBD | Bus agent |
| TBD | Pedestrian agent + safety shield (safety guard) |
| TBD | Incident agent + anomaly detection |

## 3. Folders

```
FlowMind/
├── README.md          # read me first
├── config.yaml        # all settings (times, thresholds)
├── requirements.txt   # libraries to install
├── scenarios/         # the street map and the traffic
├── flowmind/
│   ├── contracts.py   # what every agent receives (shared by all)
│   ├── env.py         # talks to SUMO (only this file does)
│   ├── control/       # manager + coordinator (Adem), safety guard (pedestrian owner)
│   ├── agents/        # one file per agent
│   ├── logger.py      # saves every decision
│   └── kpis.py        # computes the results
├── scripts/           # run.py, train.py, evaluate.py
├── sandbox/           # Ahmed's first demos
└── runs/              # results (not on GitHub)
```

**You only work in your own file.** For example, the bus person only edits `flowmind/agents/bus.py`.

## 4. How a decision happens (every 5 seconds)

1. We read what's happening at each junction.
2. Each agent says if it wants control ("a bus is 60 s late, I want control").
3. The manager picks one agent. Priority: **emergency → incident → pedestrian → bus → flow**.
4. That agent chooses which direction gets green.
5. The safety guard checks it. If it's unsafe, it changes it and writes why.
6. The light changes (yellow and red are added automatically).
7. The decision is saved in a file.

## 5. What your agent receives

Every 5 seconds, your agent gets this information about its junction:

| Information | Example |
|---|---|
| Current green direction and for how long | direction 2, for 14 s |
| Cars waiting on each lane | north: 8, east: 2 |
| Waiting time on each lane | north: 95 s |
| Buses and ambulances nearby | bus 12, 120 m away, 70 s late |
| Pedestrians waiting at each crossing | 4 people, longest wait 80 s |
| Warnings from the coordinator | ambulance arriving in 18 s |

## 6. How to write your agent

Your agent is one file with **3 functions**:

| Function | Question it answers | Example (bus) |
|---|---|---|
| `wants_control` | Do I need control now? | "Yes: a bus 70 s late is 120 m away" |
| `act` | Which direction gets green? | "The bus's direction" |
| `explain` | Why? (one sentence) | "Green for late bus 12" |

Steps:

1. Look at `agents/baselines.py`: it's a finished example.
2. Write your 3 functions, starting with simple rules.
3. Put your numbers (60 s, 120 m...) in `config.yaml`, not in the code.
4. Watch it work: `python scripts/run.py --gui`
5. Your result = your agent vs fixed-timer lights.

## 7. Install and run

```bash
git clone https://github.com/valentino3015/5DS5_FlowMind_SUMO.git
cd 5DS5_FlowMind_SUMO
git checkout dev
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

python scripts/run.py --gui     # watch the simulation
```

## 8. Plan

| When | Adem | Everyone else |
|---|---|---|
| **Thu 15 – Fri 16** | SUMO connection + street map working | Everyone writes their agent's rules on paper |
| **Sat 17 – Mon 19** | Manager working, with a basic safety guard | Code your agent; pedestrian owner adds the safety guard rules; flow agent starts learning |
| **Tue 20 – Wed 21** | Coordinator working | Results script; test agents in the simulation |
| **Thu 22 – Fri 23** | Fix bugs, prepare demo | Run all tests (10 simulations each), charts |
| **Sat 24** | **Everything works. No new features after this.** | |
| **Sun 25 – Wed 28** | Report, slides, practice the demo | Validation Wed 28 |

## 9. Rules

1. Work only in your own file.
2. Don't change `contracts.py` without telling everyone.
3. Use Git branches: `yourname/topic` → pull request to `dev`.
4. Don't upload `runs/` or big files to GitHub.
5. After Sat 24: only bug fixes.

---

## Présentation initiale conservée depuis main

# Intelligent Multi-Agent Traffic Management

University project (Design Thinking): adaptive traffic-signal control with reinforcement-learning agents, built on [SUMO](https://eclipse.dev/sumo/) and [SUMO-RL](https://github.com/LucasAlegre/sumo-rl).

## Overview
Each traffic light is an independent Q-learning agent. It observes lane density and queues, chooses the next green phase, and is rewarded for reducing waiting time. We compare the agents against a **fixed-time** baseline under different traffic conditions.

## Features
- Single-intersection and multi-intersection scenarios (3-intersection Cologne map, 4x4 grids)
- Fixed-time baseline vs. Q-learning agents
- Comparison of reward functions: `diff-waiting-time`, `queue`, `average-speed`, `pressure`
- Metrics: mean waiting time, mean speed, stopped vehicles
- Auto-generated plots, CSVs and summary tables in `outputs/`

## Requirements
- Python 3.10+
- [SUMO](https://sumo.dlr.de/docs/Downloads.php) installed, with the `SUMO_HOME` environment variable set (the scripts try to detect it automatically)

## Installation
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install sumo-rl traci sumolib stable-baselines3 matplotlib pandas
```

## Usage

**Single intersection**
```bash
python demo.py --episodes 5 --seconds 1800
python demo.py --gui --episodes 2 --seconds 900
```

**Multiple intersections**
```bash
python multi_demo.py --list
python multi_demo.py --scenario cologne3 --episodes 5 --seconds 1800
python multi_demo.py --scenario grid4x4 --route 4x4c1.rou.xml
python multi_demo.py --scenario cologne3 --rewards diff-waiting-time queue pressure
```

| Option | Description |
|---|---|
| `--scenario` | `cologne3`, `grid4x4`, `arterial4x4`, `single` (`multi_demo.py` only) |
| `--route` | Route (demand) file inside the scenario folder (`multi_demo.py` only) |
| `--rewards` | One or more reward functions to compare (`multi_demo.py` only) |
| `--episodes` | Training episodes (default 5) |
| `--seconds` | Simulated seconds per episode |
| `--gui` | Open `sumo-gui` |
| `--list` | List available networks (`multi_demo.py` only) |

## Results
Plots and CSVs are saved to `outputs/` (`*_comparison.png`, `*_learning_curve.png`, `*_summary.csv`).

## Limitations and future work
- Agents are independent: no communication or cooperation between intersections
- Tabular Q-learning scales poorly to large grids; deep RL (PPO/DQN) is the next step
- Single demand pattern and seed per run
- Planned extensions: emergency-vehicle priority, pedestrians, pollution-aware rewards, multi-agent cooperation

## Team
- Ahmed Abid
- Adem Dhouib
- Donia Hali
- Hadyle Zribi
- Omar Zalila
- Manoubi Derouich Rafrafi
- Firas Ben Achour
