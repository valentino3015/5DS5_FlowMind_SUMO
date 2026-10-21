# scenarios/

Maps and traffic used by `scripts/run.py --scenario NAME`.

| Folder / files | What it is | Owner |
|---|---|---|
| `corridor/` | **Our demo street:** 3 traffic lights J1-J2-J3 on a main road (2 lanes per direction) with side streets and pedestrian crossings. Random cars (flows), a bus line, pedestrians. Fixed events in every run: ambulance `amb_1` eastbound at t = 120 s, an accident blocking both eastbound lanes between J1 and J2 from about t = 700 s for 5 min, ambulance `amb_2` westbound at t = 1100 s. | Adem |
| `3x3grid/`, `cologne3.*` | Cars only (from SUMO-RL). Good for training the flow agent. | Ahmed |

To change the corridor: edit `corridor/corridor.nod.xml` or `corridor.edg.xml`, then run
`python scenarios/build.py` to regenerate `corridor.net.xml`. Traffic is in `corridor/corridor.rou.xml`
(all flows use `probability`, so each `--seed` gives different traffic).
