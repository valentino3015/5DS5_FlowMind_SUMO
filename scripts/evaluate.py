"""Run paired evaluation trials and export CSV/PNG reports."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowmind.evaluation import CONTROLLERS, SCENARIOS, export_results, run_experiment
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=["fixed_time", "rules"])
    parser.add_argument("--scenario", choices=SCENARIOS, default="3x3grid")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--model-dir", type=Path, help="Evaluate a saved emission model without replacing the active model")
    args = parser.parse_args()
    config = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8"))
    seeds = args.seeds or config["evaluation"]["seeds"]
    for seed in dict.fromkeys(seeds):
        for controller in dict.fromkeys(args.controllers):
            print(f"Running {controller} / {args.scenario} / seed={seed} / {args.seconds}s", flush=True)
            result = run_experiment(controller, args.scenario, args.seconds, seed, args.gui,
                                    model_directory=args.model_dir)
            print(f"Saved {result['run_id']} : {result['status']}", flush=True)
    comparisons = export_results()
    print(f"Reports: runs/reports/ ({len(comparisons)} comparison rows)")


if __name__ == "__main__":
    main()
