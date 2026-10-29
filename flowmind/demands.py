"""Varied traffic demands for emission RL: separated train/validation/test pools.

The generated route files are listed in ``index.json`` with their role and
SHA-256. Training refuses a directory whose files are unlisted, modified or
reserved for another role, so test demands cannot leak into learning.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from flowmind.env import ROOT

DEMAND_DIR = ROOT / "scenarios" / "3x3grid" / "demand"
ROLES = ("train", "validation", "test")
INDEX = "index.json"


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_index(root=DEMAND_DIR) -> dict:
    path = Path(root) / INDEX
    if not path.exists():
        raise ValueError(f"Index des demandes absent : {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_demands(directory, role) -> list[Path]:
    """Route files of one role, checked against the index of the parent folder."""
    if role not in ROLES:
        raise ValueError(f"Rôle de demande inconnu : {role}")
    directory = Path(directory).resolve()
    root = directory.parent
    entries = {(root / e["file"]).resolve(): e for e in load_index(root)["files"]}
    files = sorted(directory.glob("*.rou.xml"))
    if not files:
        raise ValueError(f"Aucune demande dans {directory}")
    for path in files:
        entry = entries.get(path.resolve())
        if entry is None:
            raise ValueError(f"Demande absente de l'index : {path.name}")
        if entry["role"] != role:
            raise ValueError(f"{path.name} est réservée au rôle {entry['role']}, pas {role}")
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Demande modifiée depuis sa génération : {path.name}")
    return files


def check_disjoint(*pools):
    """No identical route file may appear in two pools."""
    seen = {}
    for index, pool in enumerate(pools):
        for path in pool or ():
            digest = file_sha256(path)
            if seen.get(digest, index) != index:
                raise ValueError(f"Demande partagée entre deux lots : {Path(path).name}")
            seen[digest] = index


def describe(paths, root=None) -> list[dict]:
    """Manifest entries: relative file and SHA-256."""
    root = Path(root or ROOT)
    return [{"file": Path(p).resolve().relative_to(root.resolve()).as_posix(), "sha256": file_sha256(p)}
            for p in paths]


def validation_cases(seeds, demands=None) -> list[tuple[int, Path | None]]:
    """Fixed (SUMO seed, route) pairs; each demand once, seeds in rotation."""
    if not demands:
        return [(seed, None) for seed in seeds]
    return [(seeds[i % len(seeds)], Path(d)) for i, d in enumerate(demands)]


def case_key(seed, route=None):
    """Key of one validation pair; plain seed when the scenario route is used."""
    return seed if route is None else f"{Path(route).name.removesuffix('.rou.xml')}@{seed}"
