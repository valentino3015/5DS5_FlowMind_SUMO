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
_Add team name and members here._
