# FlowMind: Commands Cheat Sheet

This document lists all executable commands for development, testing, running simulations, training the Flow agent (PPO & DQN), and switching scenarios (`3x3grid` and `cologne3`) in the FlowMind repository.

---

## 1. Environment Setup

Run these commands when setting up or updating your virtual environment.

### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

# (Optional) Install PyTorch with CUDA for NVIDIA GPU support (RTX 3070, etc.)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Linux / macOS (Bash)
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Automated Testing (`pytest`)

Use `pytest` to verify contracts, agents, and the SUMO environment.

```powershell
# Run all tests in the repository (23 passing tests)
python -m pytest

# Run with verbose output showing individual test names
python -m pytest -v

# Run only Flow agent unit tests (includes spillback-awareness tests)
python -m pytest tests/test_flow.py -v

# Run only SUMO environment integration tests
python -m pytest tests/test_env.py -v

# Run only shared contracts tests
python -m pytest tests/test_contracts.py -v
```

---

## 3. Running Simulations (`scripts/run.py`)

Run closed-loop traffic light simulations on the **3x3 Grid** (default, 9 junctions) or the **Cologne 3 corridor**.

### Default: 3x3 Grid (9 Intersections)
```powershell
# Run Flow Agent on 3x3 grid (uses trained model if available)
python scripts/run.py --controller flow

# Watch cars and lights live in sumo-gui on 3x3 grid
python scripts/run.py --controller flow --gui

# Run Flow Agent with pure starvation-aware heuristic (no checkpoint loaded)
python scripts/run.py --controller flow --heuristic --gui

# Run baseline controllers on 3x3 grid
python scripts/run.py --controller fixed_time --gui
python scripts/run.py --controller max_pressure --gui
```

### Switching Scenarios (`--scenario`)
```powershell
# Explicitly run on the 3x3 Grid
python scripts/run.py --scenario 3x3grid --controller flow --gui

# Run on Cologne 3-intersection corridor
python scripts/run.py --scenario cologne3 --controller flow --gui
```

### Custom Duration & Random Seed
```powershell
# Run 300 seconds of simulation with seed 101
python scripts/run.py --controller flow --seconds 300 --seed 101

# Run for a full 1-hour simulation (3600 seconds)
python scripts/run.py --controller flow --seconds 3600 --seed 100

# View all available CLI flags and options
python scripts/run.py --help
```

---

## 4. Deep Reinforcement Learning Training (`scripts/train.py`)

Train the Flow agent with Deep RL (**PPO** or **DQN**) using `stable-baselines3`.

Features included during training:
- **Box-blocking penalties** (heavy penalty for vehicles trapped in the intersection).
- **Spillback signals** (downstream link occupancies encoded in state space).
- **Dynamic Red Clearance Extension** (prevents green into an occupied intersection box).

### PPO Training
```powershell
# Train PPO on the 3x3 grid (default bottleneck junction '4')
python scripts/train.py --algo ppo --timesteps 50000 --seed 100

# Quick test run (5,000 timesteps)
python scripts/train.py --algo ppo --timesteps 5000 --seed 100

# Train PPO on the Cologne corridor
python scripts/train.py --scenario cologne3 --algo ppo --timesteps 5000 --seed 100
```

### Deep Q-Network (DQN) Training
```powershell
# Train DQN with experience replay buffer on 3x3 grid
python scripts/train.py --algo dqn --timesteps 25000 --seed 100
```

### Hardware Device Selection (`--device`)
```powershell
# Auto-detect best available device (default)
python scripts/train.py --device auto --timesteps 5000

# Force CPU execution (recommended for single-agent tabular RL rollouts)
python scripts/train.py --device cpu --timesteps 5000

# Force NVIDIA CUDA execution (requires PyTorch CUDA build)
python scripts/train.py --device cuda --timesteps 5000
```

*Models are automatically saved to `runs/models/flow_ppo.zip` upon completion or upon pressing `Ctrl+C`.*

---

## 5. Evaluation & KPIs (`scripts/evaluate.py`)

Benchmark controllers (`fixed_time`, `max_pressure`, `flow_heuristic`, `flow`) head-to-head across seeds and automatically generate reports for all 5 core transportation engineering KPIs:
1. **Total vehicles that arrived at their destinations** (throughput & completion rate)
2. **Average and total waiting time** (stopped delay)
3. **Average travel time** (completed trip durations)
4. **Average speed over the entire run** (m/s and km/h)
5. **Cumulative halting and box-blocking delay** (veh·s and intersection spillback incidents)

```powershell
# Run benchmark on 3x3 grid (comparing fixed_time, flow_heuristic, flow)
python scripts/evaluate.py --scenario 3x3grid --seconds 600 --seeds 100

# Benchmark across multiple seeds with all controllers
python scripts/evaluate.py --scenario 3x3grid --controllers fixed_time max_pressure flow_heuristic flow --seconds 600 --seeds 100 101 102

# Benchmark on Cologne corridor
python scripts/evaluate.py --scenario cologne3 --seconds 600 --seeds 100

# Results are saved as CSV to runs/evaluation_<scenario>.csv and printed in the terminal
```

---

## 6. Sandbox Demos (`sandbox/`)

These are the initial standalone exploratory scripts using `sumo-rl`.

### Single Intersection Demo
```powershell
# Run headless single-intersection demo
python sandbox/Demo.py

# Run with sumo-gui
python sandbox/Demo.py --gui

# Run shorter episode (3 episodes, 1800s each)
python sandbox/Demo.py --episodes 3 --seconds 1800
```

### Multi-Intersection Demo
```powershell
# List available networks and route files
python sandbox/Multi_Demo.py --list

# Run Cologne 3-intersection benchmark (headless, 5 episodes)
python sandbox/Multi_Demo.py --scenario cologne3 --episodes 5 --seconds 1800

# Run Cologne with sumo-gui
python sandbox/Multi_Demo.py --scenario cologne3 --gui --episodes 2 --seconds 900
```
