# Plan: Train DQN with SB3 (mirroring d3rlpy config)

## Context
The user has a working d3rlpy DQN trained on `SalesNegotiationEnv` and wants an equivalent SB3 DQN for comparison. The goal is to match hyperparameters as closely as possible.

## Parameter Mapping

| d3rlpy | Value | SB3 equivalent | Value |
|---|---|---|---|
| `learning_rate` | 2e-5 | `learning_rate` | 2e-5 |
| `gamma` | 0.99 | `gamma` | 0.99 |
| `target_update_interval` | 20000 | `target_update_interval` | 20000 |
| `batch_size` | 256 | `batch_size` | 256 |
| `hidden_units=[128,64]` | encoder | `policy_kwargs={"net_arch": [128,64]}` | network |
| `FIFOBuffer(limit=50000)` | buffer | `buffer_size` | 50000 |
| `n_steps=400000` | total | `total_timesteps` | 400000 |
| `start_epsilon=1.0, end=0.05, duration=100000` | exploration | `exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.25` | 100000/400000 |
| `update_interval=4` | train freq | `train_freq` | 4 |
| `update_start_step=40000` | warmup | `learning_starts` | 40000 |
| `MinMaxRewardScaler(-10,10)` | reward scaling | Custom `RewardScaleWrapper` | (r - min)/(max - min)*2 - 1 |
| tensorboard + file logging | logging | `tensorboard_log="./sb3_logs/"` | tensorboard |

## Implementation Steps

### Step 1: Create `training.ipynb` with the following cells

**Cell 1 — Imports & Setup**
- Import SB3 DQN, gymnasium, gym_env, torch, numpy

**Cell 2 — MinMax Reward Wrapper**
- Define a `gymnasium.RewardWrapper` that scales rewards from [-10, 10] to [-1, 1] (no built-in SB3 equivalent)

**Cell 3 — Environment Creation**
- Create training env and eval env, wrap both with the reward scaler

**Cell 4 — DQN Model Setup**
- Instantiate DQN with all mapped parameters
- Set tensorboard logging to `./sb3_logs/`

**Cell 5 — Training**
- Add `EvalCallback` for periodic evaluation
- Call `model.learn(total_timesteps=400000)`

**Cell 6 — Save Model**
- Save to `./models/dqn_sales_negotiation.zip`

### Files to create/modify
- `training.ipynb` — new notebook with training workflow
- `main.py` — left as-is (placeholder)

## Verification
- `uv run python main.py` — should train for 400k steps with tensorboard logging
- `tensorboard --logdir sb3_logs/` — verify reward curves appear
