import matplotlib.pyplot as plt
import numpy as np


def running_mean(data, window_size):
    """
    Compute the running mean of a 1D array.

    Args:
        data (list): Input data.
        window_size (int): Window size for the running mean.

    Returns:
        list: Running mean of the input data.
    """

    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_running_average(q_learning_rewards, sarsa_rewards, window_size=100):
    """
    Plot the running average of cumulative rewards for Q-learning and SARSA.

    Args:
        q_learning_rewards (list): Cumulative rewards for Q-learning.
        sarsa_rewards (list): Cumulative rewards for SARSA.
        window_size (int): Window size for computing the running average.
    """

    q_learning_running_avg = running_mean(q_learning_rewards, window_size)
    sarsa_running_avg = running_mean(sarsa_rewards, window_size)

    plt.figure(figsize=(12, 6))
    plt.plot(q_learning_running_avg, label="Q-learning")
    plt.plot(sarsa_running_avg, label="SARSA")
    plt.xlabel("Episode")
    plt.ylabel("Running Average Cumulative Reward")
    plt.title(f"Q-learning vs SARSA: Running Average (Window Size = {window_size})")
    plt.legend()
    plt.grid(True)
    plt.savefig("q_learning_vs_sarsa_running_avg.png")
    plt.show()


def plot_zoomed_rewards(q_learning_rewards, sarsa_rewards, zoom_start, zoom_end):
    """
    Plot the zoomed average cumulative rewards for Q-learning and SARSA over episodes.

    Args:
        q_learning_rewards (list): Average cumulative rewards for Q-learning.
        sarsa_rewards (list): Average cumulative rewards for SARSA.
        zoom_start (int): Start episode for zooming.
        zoom_end (int): End episode for zooming.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(
        range(zoom_start, zoom_end),
        q_learning_rewards[zoom_start:zoom_end],
        label="Q-learning",
    )
    plt.plot(
        range(zoom_start, zoom_end), sarsa_rewards[zoom_start:zoom_end], label="SARSA"
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.title(
        f"Q-learning vs SARSA: Average Cumulative Rewards (Episodes {zoom_start}-{zoom_end})"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig("zoomed_q_learning_vs_sarsa_rewards.png")
    plt.show()
