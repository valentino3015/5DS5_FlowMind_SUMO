"""Repeated paired emission tests; preserve every run and simulator variability."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowmind.evaluation import run_experiment, export_results
from flowmind.kpis import atomic_json, write_csv

KEYS = ('co2_kg', 'nox_g', 'fuel_kg', 'pmx_g')


def summarize(rows, candidate_controller="emission_model"):
    pairs = {}
    for row in rows:
        pairs.setdefault((row['seed'], row['repeat']), {})[row['controller']] = row
    if not pairs or any(set(pair) != {'fixed_time', candidate_controller} for pair in pairs.values()):
        raise ValueError('Comparaisons incomplètes')
    if any(pair['fixed_time'][key] != pair[candidate_controller][key]
           for pair in pairs.values() for key in ('input_hash', 'protocol_hash', 'seconds', 'scenario')):
        raise ValueError('Conditions de comparaison différentes')
    metrics = []
    for key in KEYS:
        reference = [pair['fixed_time'][key] for pair in pairs.values()]
        candidate = [pair[candidate_controller][key] for pair in pairs.values()]
        spreads = []
        separated = True
        for seed in sorted({r['seed'] for r in rows}):
            groups = {c: [r[key] for r in rows if r['seed'] == seed and r['controller'] == c]
                      for c in ('fixed_time', candidate_controller)}
            separated &= max(groups[candidate_controller]) < min(groups['fixed_time'])
            for values in groups.values():
                spreads.append(100 * (max(values) - min(values)) / statistics.mean(values))
        metrics.append({'metric': key, 'fixed_time_mean': statistics.mean(reference),
                        ('rl_mean' if candidate_controller == 'emission_model' else 'rules_mean'): statistics.mean(candidate),
                        'paired_change_percent': statistics.mean(100 * (b/a-1) for a, b in zip(reference, candidate)),
                        'largest_within_seed_range_percent': max(spreads),
                        'all_seed_ranges_separated_below': bool(separated)})
    exact = all(len({r['sample_sha256'] for r in rows if r['seed'] == seed and r['controller'] == controller}) == 1
                for seed in {r['seed'] for r in rows} for controller in ('fixed_time', candidate_controller))
    traffic = all(pair[candidate_controller]['departed'] >= pair['fixed_time']['departed'] and
                  pair[candidate_controller]['arrived'] >= pair['fixed_time']['arrived'] for pair in pairs.values())
    safety = all(r['teleport_events'] == r['collision_vehicle_events'] == 0 for r in rows)
    return {'metrics': metrics, 'exact_samples_reproducible': exact,
            'no_traffic_shortfall_each_pair': traffic, 'no_collisions_or_teleports': safety,
            'clear_four_metric_gain_in_this_sample': all(m['all_seed_ranges_separated_below'] for m in metrics) and traffic and safety}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', choices=['emission_model', 'rules'], default='emission_model')
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.controller == 'emission_model' and args.model_dir is None:
        p.error('--model-dir est requis pour le modèle RL')
    if args.repeats < 2 or len(set(args.seeds)) != len(args.seeds):
        p.error('Répétitions >=2 et seeds uniques requises')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Choisir un nouveau dossier de sortie')
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(1)
    rows = []
    for seed in args.seeds:
        for repeat in range(args.repeats):
            # Alternate execution order to reduce drift linked to run order.
            order = ['fixed_time', args.controller] if repeat % 2 == 0 else [args.controller, 'fixed_time']
            for controller in order:
                row = run_experiment(controller, '3x3grid', args.seconds, seed, model_directory=args.model_dir)
                row['repeat'] = repeat
                row['sample_sha256'] = hashlib.sha256((ROOT/'runs'/row['run_id']/'timeseries.csv').read_bytes()).hexdigest()
                rows.append(row)
                atomic_json(args.output/'trials.json', {'rows': rows})
                print(seed, repeat, controller, {k: row[k] for k in KEYS}, flush=True)
    report = summarize(rows, args.controller)
    report.update(candidate_controller=args.controller, model_directory=str(args.model_dir.resolve()) if args.model_dir else None, seeds=args.seeds, repeats=args.repeats,
                  seconds=args.seconds, activated=False)
    atomic_json(args.output/'comparison.json', report)
    write_csv(args.output/'comparaison.csv', report['metrics'])
    export_results()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
