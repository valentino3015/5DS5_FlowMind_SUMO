"""Held-out validation probe: sample the learned PPO policy reproducibly."""
from pathlib import Path
import sys, json, copy, argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml, torch, numpy as np
from stable_baselines3 import PPO
from scripts.train_emissions import EmissionTrainingEnv, official_emissions, compare_validation
from flowmind.evaluation import SCENARIOS, digest_files, emission_protocol
from flowmind.kpis import atomic_json

p=argparse.ArgumentParser()
p.add_argument('--model-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint',choices=['best','last'],default='best')
p.add_argument('--baseline',type=Path,required=True,help='validation_baseline.json from the matching scenario and duration')
p.add_argument('--seeds',type=int,nargs='+',help='Held-out validation seeds; defaults to seeds in the baseline')
args=p.parse_args()
torch.set_num_threads(1)
manifest=json.loads((args.model_dir/'manifest.json').read_text(encoding='utf-8'))
config=emission_protocol(yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8')))
net,route=SCENARIOS[manifest['scenario']]
config['sim'].update(scenario=manifest['scenario'],net_file=net,route_file=route,seconds=manifest['seconds'])
baseline=json.loads(args.baseline.read_text(encoding='utf-8'))
baseline={int(k):v for k,v in baseline['seeds'].items()}
model=PPO.load(args.model_dir/f'{args.checkpoint}.zip',device='cpu')
measured,stats,scores={},{},[]
if args.output.exists() and any(args.output.iterdir()):
    p.error('Output must be new or empty')
args.output.mkdir(parents=True,exist_ok=True)
seeds=args.seeds or sorted(baseline)
if any(seed not in baseline or seed in manifest['train_seeds'] for seed in seeds):
    p.error('Use baseline validation seeds distinct from training seeds')
for seed in seeds:
    directory=args.output/f'seed_{seed}'
    cfg=copy.deepcopy(config);cfg['evaluation_output']=str(directory.resolve())
    env=EmissionTrainingEnv(cfg,manifest,[seed])
    model.set_random_seed(100000+seed)
    try:
        obs,_=env.reset()
        finished=False
        while not finished:
            action,_=model.predict(obs,deterministic=False)
            obs,_,terminated,truncated,_=env.step(action)
            finished=terminated or truncated
        scores.append(env.total_reward)
        stats[seed]=dict(env.episode_stats)
    finally:
        env.close()
    measured[seed]=official_emissions(directory/'tripinfo.xml')
    print(f'Sampling validation seed {seed}: {measured[seed]}, {stats[seed]}',flush=True)
compared=compare_validation(measured,baseline)
result={'sampling':'PPO learned categorical probabilities','policy_seed_offset':100000,
        'model':str(args.model_dir.resolve()),'checkpoint':args.checkpoint,
        'checkpoint_timesteps':model.num_timesteps,
        'checkpoint_hash':digest_files([args.model_dir/f'{args.checkpoint}.zip']),
        'comparison':compared,'reward':float(np.mean(scores)),
        'teleports':sum(s['teleports'] for s in stats.values()),
        'collisions':sum(s['collisions'] for s in stats.values()),'measured':measured,'stats':stats}
atomic_json(args.output/'result.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
