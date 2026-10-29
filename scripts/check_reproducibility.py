"""Compare complete samples and decisions across fresh Python/SUMO processes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', choices=['fixed_time', 'emission_model'], default='fixed_time')
    p.add_argument('--seed', type=int, default=1004)
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--count', type=int, default=3)
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker is not None:
        import torch
        from flowmind.evaluation import run_experiment
        torch.set_num_threads(1)
        result = run_experiment(args.controller, '3x3grid', args.seconds, args.seed,
                                model_directory=args.model_dir)
        directory = ROOT / 'runs' / result['run_id']
        hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                  for name in ['timeseries.csv', 'decisions.csv']}
        from scripts.train_emissions import official_emissions
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f'{args.worker}.json').write_text(json.dumps({
            'result': result, 'hashes': hashes,
            'official': official_emissions(directory / 'tripinfo.xml')}, indent=2), encoding='utf-8')
        return
    if args.count < 2:
        p.error('Au moins deux répétitions nécessaires')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Choisir un nouveau dossier de sortie')
    rows = []
    for i in range(args.count):
        command = [sys.executable, str(Path(__file__).resolve()), '--controller', args.controller,
                   '--seed', str(args.seed), '--seconds', str(args.seconds),
                   '--output', str(args.output.resolve()), '--worker', str(i)]
        if args.model_dir:
            command += ['--model-dir', str(args.model_dir.resolve())]
        subprocess.run(command, check=True, cwd=ROOT,
                       env=dict(os.environ, PYTHONHASHSEED=str(i)))
        rows.append(json.loads((args.output / f'{i}.json').read_text(encoding='utf-8')))
        print(f'Répétition {i+1}/{args.count} terminée', flush=True)
    passed = all(row['hashes'] == rows[0]['hashes'] and row['official'] == rows[0]['official']
                 and row['result']['protocol_hash'] == rows[0]['result']['protocol_hash'] for row in rows)
    report = {'passed': passed, 'fresh_processes': True, 'different_python_hash_seeds': True,
              'seed': args.seed, 'seconds': args.seconds, 'controller': args.controller, 'runs': rows}
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Reproductibilité exacte :', passed, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
