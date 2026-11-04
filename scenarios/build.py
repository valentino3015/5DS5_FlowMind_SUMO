"""Build a corridor street map: NAME.nod.xml + NAME.edg.xml -> NAME.net.xml.

Owner: Adem.

Run once after editing the node or edge files:
    python scenarios/build.py              # scenarios/corridor/  (3 junctions)
    python scenarios/build.py corridor5    # scenarios/corridor5/ (5 junctions)
netconvert adds sidewalks and pedestrian crossings automatically.
"""
import subprocess
import sys
from pathlib import Path

import sumo


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "corridor"
    here = Path(__file__).parent / name
    netconvert = Path(sumo.SUMO_HOME) / "bin" / "netconvert"
    cmd = [
        str(netconvert),
        "--node-files", str(here / f"{name}.nod.xml"),
        "--edge-files", str(here / f"{name}.edg.xml"),
        "--output-file", str(here / f"{name}.net.xml"),
        "--sidewalks.guess",          # add a sidewalk to every road
        "--crossings.guess",          # add pedestrian crossings at the junctions
        "--tls.default-type", "static",
        "--no-turnarounds",           # cars do not make U-turns
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote {here / f'{name}.net.xml'}")


if __name__ == "__main__":
    main()
