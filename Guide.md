# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

FlowMind: traffic-signal control in SUMO for a 3-junction corridor. Per junction, a rule-based orchestrator hands control to one specialist agent (flow = PPO reinforcement learning; emergency, bus, pedestrian, incident = rules); a rule-based safety shield checks every action; a rule-based corridor coordinator posts events across junctions.

**Hard deadline: feature freeze Sat 24 Oct 2026, validation Wed 28 Oct.** A team of 7 students must understand and present every file. Read `README.md` first (structure and plan, written in plain language for the team). The shared data types are defined in `flowmind/contracts.py`.

## Ownership

Adem: `contracts.py`, `env.py`, `logger.py`, `scenarios/`, `control/orchestrator.py`, `control/coordinator.py`.
Teammates: `control/shield.py` (pedestrian agent owner), each file in `agents/`, `kpis.py` and `scripts/evaluate.py` (evaluation owner), `scripts/train.py` (flow owner).
When working on Adem's files, do not rewrite teammates' files; leave clear hooks and TODOs for them instead.

## Hard rules

1. **Simple beats clever.** Plain classes and functions, no framework layers, no abstract hierarchies beyond `Specialist`. If a file passes ~300 lines, ask whether it can be simpler.
2. **Never use SUMO-RL's multi-agent environment** in `flowmind/`. SUMO-RL may appear only in `sandbox/`.
3. **No LLM calls** anywhere in `flowmind/`.
4. **Only `flowmind/env.py` imports `traci`/`libsumo`.** `control/` and `agents/` import only `flowmind.contracts` (plus numpy and the standard library).
5. **`contracts.py` is shared by the whole team.** Do not change it unless the task says so; if it seems needed, stop and explain.
6. **Agents return a green phase index only.** Yellow (3 s) and all-red (2 s) are inserted by `env.py`.
7. **No magic numbers:** parameters live in `config.yaml`.
8. **Windows compatible:** `pathlib`, no shell-specific commands in Python.
9. Do not commit `runs/`, generated route files or model files.

## Stack

Python 3.10+, `eclipse-sumo`, `traci`, `libsumo`, `pettingzoo`, `gymnasium`, `stable-baselines3`, `supersuit`, `numpy`, `pandas`, `pyyaml`, `matplotlib`, `pytest`.

## Commands

```bash
pip install -r requirements.txt
python scripts/run.py --controller fixed_time [--gui] [--seed N] [--scenario NAME]
python scripts/run.py --controller orchestrated [--gui]
python scripts/train.py
python scripts/evaluate.py
pytest
```

## Working style

- Work in the phases given in the prompt; at the end of each phase run `pytest` and a short simulation, then stop for review.
- When something is not specified, choose the simplest option and say so in your summary.
