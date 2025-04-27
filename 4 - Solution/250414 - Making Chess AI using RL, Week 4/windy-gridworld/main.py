from assignment import q_learning, sarsa, expected_sarsa
from environment import WindyGridworld
from visualizations import plot_running_average, plot_zoomed_rewards


def main():

    # Define the windy gridworld environment
    width, height = 10, 7
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    start_state = (0, 3)
    goal_state = (7, 3)

    env = WindyGridworld(width, height, wind, start_state, goal_state)

    # Run Q-learning and SARSA algorithms
    q_learning_rewards = q_learning(env, episodes=550, trials=1000)
    sarsa_rewards = sarsa(env, episodes=550, trials=1000)

    # Visualize the running average of cumulative rewards
    plot_running_average(q_learning_rewards, sarsa_rewards)

    # Zoom in on the area where Q-learning shows superiority over SARSA
    zoom_start = 350
    zoom_end = 550
    plot_zoomed_rewards(q_learning_rewards, sarsa_rewards, zoom_start, zoom_end)


if __name__ == "__main__":
    main()
