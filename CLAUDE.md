# DQN_SB3 - Sales Negotiation RL Project

## Project Overview
Reinforcement learning project using Stable Baselines 3 (SB3) to train agents on a custom Sales Negotiation Gymnasium environment.

## Tech Stack
- Python 3.11+, managed with `uv`
- Gymnasium for the RL environment
- Stable Baselines 3 (SB3) for RL algorithms
- PyTorch (CUDA 12.1) as the backend

## Environment: SalesNegotiationEnv (`gym_env.py`)
- **Goal:** Agent is a salesperson negotiating with a customer who has objections (A, B, C, D).
- **Latent variables:** Customer Desire (D ∈ [0.8,1]) and Patience (P ∈ [0.6,0.9]), randomized per episode.
- **Actions (3):** Persuade (0), Incentive (1), Close (2).
- **Objections:** A, B, C are resolvable; D is infinite/generic.
- **Incentive costs:** A=2, B=5, C=7. One use per objection topic. Not valid on D.
- **Rewards:** Sale = +10, labor cost = -0.2/round, exit penalty = -2.
- **Termination:** Sale, close, customer exit (after round 5), rule violation, or max 30 rounds.
- **Observation space:** MultiDiscrete — round, resolved flags, incentive used, prev objection resolved, action history, topic history.

## Training Setup (`training.ipynb`)
DQN trained via SB3, mirroring a d3rlpy baseline for comparison.

**Key hyperparameters:**
- `learning_rate=2e-5`, `gamma=0.99`, `batch_size=256`
- `buffer_size=50000`, `total_timesteps=400000`
- `target_update_interval=20000`, `train_freq=4`, `learning_starts=40000`
- `exploration_fraction=0.25` (eps 1.0→0.05 over 100k steps)
- Network: `net_arch=[128, 64]`
- Custom `RewardScaleWrapper` maps rewards from [-10, 10] to [-1, 1]
- TensorBoard logging to `./sb3_logs/`
- Model saved to `./models/dqn_sales_negotiation.zip`

## Key Files
- `gym_env.py` — Custom Gymnasium environment (SalesNegotiationEnv)
- `training.ipynb` — DQN training notebook (SB3)
- `main.py` — Entry point (currently placeholder)
- `plan.md` — Parameter mapping and implementation plan (d3rlpy → SB3)
- `pyproject.toml` — Dependencies and project config

## Directories
- `sb3_logs/` — TensorBoard training logs
- `models/` — Saved model checkpoints

## Commands
- `uv run python main.py` — Run the main script
- `uv sync` — Install/sync dependencies
- `tensorboard --logdir sb3_logs/` — View training curves
