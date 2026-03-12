# Sales Negotiation RL

Reinforcement learning agents (DQN & A2C) trained with Stable Baselines 3 on a custom sales negotiation Gymnasium environment.

## Scenario

An RL agent acts as a salesperson negotiating with a simulated customer. The customer holds up to four objection topics (A-D), each with hidden desire and patience levels that vary per episode. The agent chooses among three actions each round — **Persuade**, **Incentive**, or **Close** — to resolve objections and close the sale. Objections A/B/C are resolvable; D is a recurring generic objection. Incentives have per-topic costs and are single-use. The episode ends on a successful sale, a failed close, customer exit (patience runs out after round 5), a rule violation, or reaching 30 rounds.

## Current Focus

**`DQN_A2C_training.ipynb`** — Main training notebook for DQN and A2C experiments. All active development and tuning happens here.

## Key Files

| File | Description |
|------|-------------|
| `DQN_A2C_training.ipynb` | Training notebook (current focus) |
| `gym_env.py` | Custom Gymnasium environment (`SalesNegotiationEnv`) |
| `utils.py` | Shared utilities |
| `main.py` | Entry point |

## Quick Start

```bash
uv sync                          # install dependencies
uv run jupyter lab                # open the training notebook
tensorboard --logdir sb3_logs/    # view training curves
```
