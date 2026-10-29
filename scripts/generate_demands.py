"""Generate varied 3x3 grid demands with SUMO randomTrips, checked with FixedTime.

Three separated pools: train, validation, test. Loads range from heavy to light
plus a peak profile; trips are local or through traffic (fringe factor). Every
file is simulated once with classical signals and regenerated with another
seed if it causes teleports or collisions. Nothing is trained here.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml
from flowmind.env import ROOT, SumoCorridorEnv, simulation_protocol
from flowmind.agents.baselines import FixedTime
from flowmind.demands import DEMAND_DIR, INDEX, ROLES, file_sha256
from flowmind.kpis import atomic_json

NET = ROOT / "scenarios" / "3x3grid" / "3x3Grid2lanes.net.xml"
# Mean seconds between insertions; several values = consecutive equal sub-intervals.
LOADS = (("p1.2", [1.2]), ("p1.6", [1.6]), ("p2.0", [2.0]),
         ("p2.5", [2.5]), ("p3.0", [3.0]), ("peak", [3.0, 1.4, 3.0]))
FRINGE_FACTORS = (1, 5, 10)
SEED_BASE = {"train": 1000, "validation": 5000, "test": 9000}
MAX_ATTEMPTS = 5


def recipe(role, index):
    """Deterministic profile of the n-th file of a pool; every load level is covered."""
    name, periods = LOADS[index % len(LOADS)]
    fringe = FRINGE_FACTORS[(index // len(LOADS)) % len(FRINGE_FACTORS)]
    return {"role": role, "index": index, "load": name, "period": periods, "fringe_factor": fringe}


def random_trips(spec, seed, seconds, output):
    tools = Path(os.environ["SUMO_HOME"]) / "tools"
    stem = output.name.removesuffix(".rou.xml")
    # Relative names: SUMO copies paths into XML comments, where "--" is illegal.
    cmd = [sys.executable, str(tools / "randomTrips.py"), "-n", str(NET),
           "-o", f"{stem}.trips.xml", "-r", output.name,
           "-b", "0", "-e", str(seconds + 60), "--seed", str(seed),
           "--fringe-factor", str(spec["fringe_factor"]),
           "--prefix", f"{spec['role'][0]}{spec['index']}_",
           "--remove-loops", "-p", *map(str, spec["period"])]
    run = subprocess.run(cmd, cwd=output.parent, capture_output=True, text=True)
    for extra in (f"{stem}.trips.xml", f"{stem}.rou.alt.xml"):
        (output.parent / extra).unlink(missing_ok=True)
    # randomTrips exits with 0 even when duarouter rejects every trip.
    if run.returncode or not output.exists() or "<vehicle" not in output.read_text(encoding="utf-8"):
        raise RuntimeError(f"randomTrips n'a produit aucun véhicule : {run.stdout[-500:]} {run.stderr[-500:]}")


def fixed_time_check(route, seconds, seed=0):
    """One classical run: teleports, collisions, departures and arrivals."""
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    config["sim"].update(seconds=seconds)
    samples = []
    env = SumoCorridorEnv(config=copy.deepcopy(config), route_file=route, seed=seed, use_gui=False)
    env.metrics_callback = samples.append
    try:
        states = env.reset()
        agents = {k: FixedTime(config["fixed_time"]) for k in states}
        while samples[-1]["elapsed_s"] < env.sim_seconds:
            states, _, _ = env.step({k: a.act(states[k], [], [True]*states[k].num_phases) for k, a in agents.items()})
    finally:
        env.close()
    return {k: int(sum(s[k] for s in samples)) for k in ("teleports", "collisions", "departed", "arrived")}


def build(spec, seconds, min_departed, output_dir):
    """Generate one file, retrying with the next seed until FixedTime runs cleanly."""
    rejected = []
    for attempt in range(MAX_ATTEMPTS):
        seed = SEED_BASE[spec["role"]] + 10 * spec["index"] + attempt
        path = output_dir / spec["role"] / f"{spec['role']}_{spec['index']:02d}_{spec['load']}_f{spec['fringe_factor']}.rou.xml"
        random_trips(spec, seed, seconds, path)
        check = fixed_time_check(path, seconds)
        if check["teleports"] == 0 and check["collisions"] == 0 and check["departed"] >= min_departed:
            return {**spec, "seed": seed, "file": path.relative_to(output_dir).as_posix(),
                    "sha256": file_sha256(path), "fixed_time": check, "rejected": rejected}
        rejected.append({"seed": seed, **check})
        path.unlink()
    raise RuntimeError(f"Aucune demande valide après {MAX_ATTEMPTS} essais : {spec}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEMAND_DIR)
    p.add_argument("--train", type=int, default=40)
    p.add_argument("--validation", type=int, default=6)
    p.add_argument("--test", type=int, default=10)
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument("--min-departed", type=int, default=50)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    args = p.parse_args()
    if "SUMO_HOME" not in os.environ:
        p.error("SUMO_HOME doit pointer vers l'installation SUMO (randomTrips.py)")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        p.error("Le dossier de sortie doit être nouveau ou vide (préserve les demandes existantes)")
    for role in ROLES:
        (output / role).mkdir(parents=True, exist_ok=True)
    specs = [recipe(role, i) for role, count in zip(ROLES, (args.train, args.validation, args.test))
             for i in range(count)]
    print(f"Génération de {len(specs)} demandes avec {args.workers} processus…", flush=True)
    files = []
    # Separate processes: each SUMO run owns its default TraCI connection.
    with ProcessPoolExecutor(args.workers) as pool:
        for entry in pool.map(build, specs, [args.seconds]*len(specs),
                              [args.min_departed]*len(specs), [output]*len(specs)):
            files.append(entry)
            print(f"{entry['file']} : seed={entry['seed']}, {entry['fixed_time']}", flush=True)
    atomic_json(output / INDEX, {
        "net": NET.relative_to(ROOT).as_posix(), "net_sha256": file_sha256(NET),
        "seconds": args.seconds, "loads": dict(LOADS), "fringe_factors": list(FRINGE_FACTORS),
        "check": "FixedTime, SUMO seed 0: no teleport, no collision, departed >= %d" % args.min_departed,
        "simulation": simulation_protocol(), "files": files})
    print(f"Index écrit : {output / INDEX}", flush=True)


if __name__ == "__main__":
    main()
