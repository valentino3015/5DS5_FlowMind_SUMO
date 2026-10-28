# Trained models

## H_h2_s0.zip — agent H (residual on rule C), current best for bunching

- **What it is:** MaskablePPO policy of agent H (`rl/h_agent.py`): rule controller C runs every
  junction; while a bus is within 150 m H chooses follow C / serve the bus / hold the bus every
  5 s, inside C's limits (green 10-60 s, extension budget, queue guard).
- **Training:** `train_ppo.py --reward regular_h --w-headway 2 --c-switch 0.5 --train-seed 0
  --steps 200000` (reward: person delay - 2 x headway deviation per bus passage - switch cost),
  SUMO training seeds 0-899, H at B1 and C elsewhere. Checkpoint `ckpt_140000.zip`;
  options in `H_h2_s0_config.json`, training-time validation in
  `H_h2_s0_training_validation.csv` (3600 s windows).
- **Why this checkpoint:** among 140k and 180k (the best ones in training), the lowest mean of
  B1 and all-9 bunching at 7200 s on validation seeds 900-904
  (`results/rl_h_val_val_*.csv`): 23.1 vs 23.8; 180k also left up to 15 cars uninserted.
- **Validation result, seeds 900-904, 7200 s (vs C at all 9; * = paired t test, 5 %):**

  | | last-stop bunching | headway CV | person delay, whole grid | person delay at B1 |
  |---|---|---|---|---|
  | H at B1, C elsewhere | 25.6 (-3.6) | 0.420 (-0.02) | +30 h* (+6 %) | +18 h* (+22 %) |
  | H at all 9 | 20.6 (-8.6*) | 0.400 (-0.04) | +61 h* (+12 %) | +19 h* (+23 %) |

  0 safety violations, 0 cars never inserted. H beats C on bunching by holding buses, at a
  clear person-delay cost. **Not tested on the evaluation seeds 10000-10019 yet**, and only one
  training seed (0).
- **Run it:**
  `python sandbox/bo4_bus/rl/evaluate.py val --scope all9 --with-b1 --prefix <name> --model H=sandbox/bo4_bus/rl/models/H_h2_s0.zip`
  (needs sb3-contrib, see `rl/requirements.txt`; the observation version is detected from the
  model's 43 inputs).
