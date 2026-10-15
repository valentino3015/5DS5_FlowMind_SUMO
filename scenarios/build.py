"""Build the corridor street map: corridor.nod.xml + corridor.edg.xml -> corridor.net.xml.

Owner: Adem.

Run once after editing the node or edge files:
    python scenarios/build.py
netconvert adds sidewalks and pedestrian crossings automatically.
"""
import subprocess
from pathlib import Path

import sumo

HERE = Path(__file__).parent / "corridor"


def main():
    netconvert = Path(sumo.SUMO_HOME) / "bin" / "netconvert"
    cmd = [
        str(netconvert),
        "--node-files", str(HERE / "corridor.nod.xml"),
        "--edge-files", str(HERE / "corridor.edg.xml"),
        "--output-file", str(HERE / "corridor.net.xml"),
        "--sidewalks.guess",          # add a sidewalk to every road
        "--crossings.guess",          # add pedestrian crossings at the junctions
        "--tls.default-type", "static",
        "--no-turnarounds",           # cars do not make U-turns
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote {HERE / 'corridor.net.xml'}")


if __name__ == "__main__":
    main()
