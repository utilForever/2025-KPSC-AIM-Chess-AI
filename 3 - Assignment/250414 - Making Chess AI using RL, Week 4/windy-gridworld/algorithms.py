import numpy as np


def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, trials=10):
    """
    Q-learning algorithm with multiple trials for averaging.

    Args:
        env (WindyGridworld): The Windy Gridworld environment.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        episodes (int): Number of episodes.
        trials (int): Number of trials to average results.

    Returns:
        avg_rewards (list): Average cumulative rewards per episode over trials.
    """

    all_rewards = np.zeros((trials, episodes))

    # TODO: Implement Q-learning algorithm

    avg_rewards = np.mean(all_rewards, axis=0)

    return avg_rewards


def sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, trials=10):
    """
    SARSA algorithm with multiple trials for averaging.

    Args:
        env (WindyGridworld): The Windy Gridworld environment.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        episodes (int): Number of episodes.
        trials (int): Number of trials to average results.

    Returns:
        avg_rewards (list): Average cumulative rewards per episode over trials.
    """

    all_rewards = np.zeros((trials, episodes))

    # TODO: Implement SARSA algorithm

    avg_rewards = np.mean(all_rewards, axis=0)

    return avg_rewards
