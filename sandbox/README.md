# sandbox/: Ahmed's SUMO-RL demos

Owner: Ahmed.

These are Ahmed's first exploration scripts with [SUMO-RL](https://github.com/LucasAlegre/sumo-rl).
They compare **fixed-time** traffic lights with a simple **Q-learning** agent.

- `Demo.py`: one intersection (SUMO-RL's `single-intersection` network).
- `Multi_Demo.py`: several intersections (`cologne3`, `grid4x4`, `arterial4x4`, `single`).

**For learning only, not for the final system.** They use SUMO-RL's multi-agent
environment, which is not allowed in `flowmind/`. The final system talks to SUMO
only through `flowmind/env.py`.

## Run

From the repository root, with the venv active. `pip install -r requirements.txt`
installs SUMO and the `sumo-rl` package (version 1.4.5, which needs `numpy<2.4`).

```bash
python sandbox/Demo.py --episodes 2 --seconds 600
python sandbox/Demo.py --gui
python sandbox/Multi_Demo.py --list
python sandbox/Multi_Demo.py --scenario cologne3 --episodes 1 --seconds 300
python sandbox/Multi_Demo.py --scenario grid4x4 --rewards diff-waiting-time queue pressure
```

Outputs (CSV files and plots) go to `sandbox/outputs/`, which is not on GitHub.
