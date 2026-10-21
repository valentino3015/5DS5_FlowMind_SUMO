"""Core idea 8: the same seed gives exactly the same decisions (reproducible replay).

Owner: Adem. Runs the real orchestrated corridor three times for 200 s (a few seconds each).
"""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(seed: int, out: Path) -> list[dict]:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run.py"), "--controller", "orchestrated",
         "--scenario", "corridor", "--seed", str(seed), "--seconds", "200", "--run-dir", str(out)],
        check=True, capture_output=True, cwd=ROOT,
    )
    with open(out / "decisions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        del row["latency_ms"]  # real computer time, not part of the decision
    return rows


def test_same_seed_same_decisions(tmp_path):
    first = run(1, tmp_path / "a")
    second = run(1, tmp_path / "b")
    other = run(2, tmp_path / "c")
    assert len(first) > 100
    assert first == second
    assert first != other
