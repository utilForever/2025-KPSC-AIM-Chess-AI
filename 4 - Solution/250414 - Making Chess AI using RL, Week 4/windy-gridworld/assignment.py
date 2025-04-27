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

    for t in range(trials):
        Q = {}
        for x in range(env.width):
            for y in range(env.height):
                Q[(x, y)] = {a: 0 for a in env.actions}

        for e in range(episodes):
            state = env.reset()
            total_reward = 0

            while state != env.goal_state:
                if np.random.rand() < epsilon:
                    action = np.random.choice(env.actions)
                else:
                    action = max(Q[state], key=Q[state].get)

                next_state, reward = env.step(state, action)
                total_reward += reward

                best_next_action = max(Q[next_state], key=Q[next_state].get)
                Q[state][action] += alpha * (
                    reward + gamma * Q[next_state][best_next_action] - Q[state][action]
                )

                state = next_state

            all_rewards[t, e] = total_reward

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

    for t in range(trials):
        Q = {}
        for x in range(env.width):
            for y in range(env.height):
                Q[(x, y)] = {a: 0 for a in env.actions}

        for e in range(episodes):
            state = env.reset()
            if np.random.rand() < epsilon:
                action = np.random.choice(env.actions)
            else:
                action = max(Q[state], key=Q[state].get)
            total_reward = 0

            while state != env.goal_state:
                next_state, reward = env.step(state, action)
                total_reward += reward

                if np.random.rand() < epsilon:
                    next_action = np.random.choice(env.actions)
                else:
                    next_action = max(Q[next_state], key=Q[next_state].get)

                Q[state][action] += alpha * (
                    reward + gamma * Q[next_state][next_action] - Q[state][action]
                )

                state, action = next_state, next_action

            all_rewards[t, e] = total_reward

    avg_rewards = np.mean(all_rewards, axis=0)

    return avg_rewards
