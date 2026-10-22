"""scripts/run.py: two runs started in the same second must not share a results folder.

Owner: Adem.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
run_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_script)


def test_two_processes_get_different_default_folders():
    a = run_script.default_run_dir("orchestrated", "corridor", "1", pid=1234)
    b = run_script.default_run_dir("orchestrated", "corridor", "1", pid=5678)
    assert a != b
    assert a.name.endswith("_seed1_p1234") and b.name.endswith("_seed1_p5678")
    assert a.parent == ROOT / "runs"


def test_default_uses_this_process_id():
    import os
    assert run_script.default_run_dir("fixed_time", "corridor", "1").name.endswith(f"_p{os.getpid()}")
