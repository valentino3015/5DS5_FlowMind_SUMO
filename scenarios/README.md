# scenarios/

Maps and traffic used by `scripts/run.py --scenario NAME` and by the dashboard.

| Folder / files | What it is | Owner |
|---|---|---|
| `corridor/corridor.rou.xml` (`corridor`) | **The scenario of all reported results (do not change it).** 3 traffic lights J1-J2-J3 on a main road (2 lanes per direction, 200 m apart) with side streets and pedestrian crossings. 30 min of random cars (flows), a bus line both ways, pedestrians. Fixed events in every run: ambulance `amb_1` eastbound at t = 120 s, an accident blocking both eastbound lanes between J1 and J2 from about t = 700 s for 5 min, ambulance `amb_2` westbound at t = 1100 s. | Adem |
| `corridor/corridor_demo.rou.xml` (`corridor_demo`) | Same street and normal traffic, 10 min where every agent acts: `amb_1` at 60 s, pedestrian rush at J2 at 140 s, accident between J1 and J2 at 230 s (4 min) with an eastbound peak, late `bus_ew.1` and bunched `bus_ew.2` at 330 s, `amb_2` at 430 s while the incident agent handles the accident. | Adem |
| `corridor5/` (`corridor5`, `corridor5_demo`) | The same design with 5 junctions J1-J5 (side streets at each one). 30-min version: `amb_1` 120 s, accident between J2 and J3 from ~700 s for 5 min, `amb_2` 1100 s. Demo: same events as `corridor_demo`, with the pedestrian rush at J3 and the accident between J2 and J3. | Adem |
| `3x3grid/`, `cologne3.*` | Cars only (from SUMO-RL). Flow-agent training and the emission studies. | Ahmed, Manoubi |

To change a corridor map: edit `NAME/NAME.nod.xml` or `NAME/NAME.edg.xml`, then run
`python scenarios/build.py` (corridor) or `python scenarios/build.py corridor5` to regenerate
`NAME.net.xml`. Traffic is in the `.rou.xml` files (all flows use `probability`, so each `--seed`
gives different traffic).

Route files must stay in departure order: SUMO silently drops an entry that starts earlier than
the one before it (`tests/test_demo_scenario.py` checks every corridor route file).
