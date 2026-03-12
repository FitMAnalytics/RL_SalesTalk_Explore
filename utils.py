import numpy as np
import torch
from tqdm import tqdm


def get_q_values_dqn(model, obs):
    """Get Q-values for all actions from an SB3 DQN model."""
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
    return q_values.cpu().numpy()[0]

def evaluate_strategy_dqn(model, env, n_episodes=100, verbose=False):
    test_record = []
    round_count_record = []
    last_action_record = []
    topic_history_record = []
    action_history_record = []

    for _ in tqdm(range(n_episodes), desc="Evaluating DQN", disable=not verbose):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        current_round = 0
        action_history = []

        while not done:
            current_round += 1
            state = env.unpack_obs(obs)
            q_values = get_q_values_dqn(model, obs)

            current_topic = state["topic_history"][state["current_round"]]
            incentive_used = state["incentive_used"]

            if current_topic == 4 or incentive_used:
                action = 0 if q_values[0] > q_values[2] else 2
            else:
                action = int(np.argmax(q_values))

            action_history.append(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        test_record.append(episode_reward)
        round_count_record.append(current_round)
        last_action_record.append(action)
        topic_history_record.append(state["topic_history"][:current_round])
        action_history_record.append(action_history)

    return test_record, round_count_record, last_action_record, topic_history_record, action_history_record



def get_action_probs_a2c(model, obs):
    """Get action probabilities from A2C policy network."""
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.squeeze(0).cpu().numpy()
    return probs  # [p_persuade, p_incentive, p_close]


def evaluate_strategy_a2c(model, env, n_episodes=100, verbose=False):
    test_record = []
    round_count_record = []
    last_action_record = []
    topic_history_record = []
    action_history_record = []

    for _ in tqdm(range(n_episodes), desc="Evaluating A2C", disable=not verbose):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        current_round = 0
        action_history = []

        while not done:
            current_round += 1
            state = env.unpack_obs(obs)
            probs = get_action_probs_a2c(model, obs)

            current_topic = state["topic_history"][state["current_round"]]
            incentive_used = state["incentive_used"]

            if current_topic == 4 or incentive_used:
                probs[1] = 0.0
                probs = probs / probs.sum()

            action = np.random.choice(len(probs), p=probs)

            action_history.append(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        test_record.append(episode_reward)
        round_count_record.append(current_round)
        last_action_record.append(action)
        topic_history_record.append(state["topic_history"][:current_round])
        action_history_record.append(action_history)

    return test_record, round_count_record, last_action_record, topic_history_record, action_history_record