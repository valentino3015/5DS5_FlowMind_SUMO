# FlowMind: Commands Cheat Sheet

This document lists all the executable commands for development, testing, running simulations, training the Flow agent, and evaluating models in the FlowMind repository.

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
# Run all tests in the repository (19 tests)
python -m pytest

# Run with verbose output showing individual test names
python -m pytest -v

# Run only Flow agent unit tests (Ahmed)
python -m pytest tests/test_flow.py -v

# Run only SUMO environment integration tests (Adem)
python -m pytest tests/test_env.py -v

# Run only shared contracts tests
python -m pytest tests/test_contracts.py -v
```

---

## 3. Running Corridor Simulations (`scripts/run.py`)

Run closed-loop simulations on the 3-junction corridor.

### Quick Runs (Headless / Fast)
```powershell
# Run with Fixed-Time baseline (timer controller)
python scripts/run.py --controller fixed_time

# Run with Max-Pressure baseline (queue difference controller)
python scripts/run.py --controller max_pressure

# Run with Flow Agent (Ahmed's queue-draining heuristic / PPO)
python scripts/run.py --controller flow
```

### Visual Runs with SUMO GUI
```powershell
# Watch cars and traffic lights live in sumo-gui (Fixed-Time)
python scripts/run.py --controller fixed_time --gui

# Watch cars and traffic lights live in sumo-gui (Max-Pressure)
python scripts/run.py --controller max_pressure --gui

# Watch cars and traffic lights live in sumo-gui (Flow Agent)
python scripts/run.py --controller flow --gui
```

### Custom Duration & Random Seed Options
```powershell
# Run for 300 seconds of simulation with seed 101
python scripts/run.py --controller flow --seconds 300 --seed 101

# Run for a full 1-hour simulation (3600 seconds)
python scripts/run.py --controller flow --seconds 3600 --seed 100

# View all available CLI flags and options
python scripts/run.py --help
```

---

## 4. Training the Flow Agent (`scripts/train.py`)

Train the PPO reinforcement learning model with `stable-baselines3`.

```powershell
# View training script help and options
python scripts/train.py --help

# Run PPO training with default settings (50,000 timesteps, seed 100)
python scripts/train.py --timesteps 50000 --seed 100

# Run a quick training check (5,000 timesteps)
python scripts/train.py --timesteps 5000 --seed 102
```

*Trained models are automatically saved to `runs/models/flow_ppo.zip` (ignored by Git).*

---

## 5. Evaluation & KPIs (`scripts/evaluate.py`)

Benchmark trained agents against baselines across multiple random seeds.

```powershell
# Run benchmark across all 10 evaluation seeds ([100..109])
python scripts/evaluate.py
```

---

## 6. Sandbox Demos (`sandbox/`)

These are the initial standalone demos built for early exploration with `sumo-rl`.

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

# Run Grid 4x4 with multiple reward functions
python sandbox/Multi_Demo.py --scenario grid4x4 --rewards diff-waiting-time queue pressure
```
