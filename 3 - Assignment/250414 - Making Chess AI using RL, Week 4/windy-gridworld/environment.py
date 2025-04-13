import numpy as np


class WindyGridworld:
    """
    Windy Gridworld environment where wind pushes the agent up in certain columns.

    Attributes:
        width (int): Width of the grid.
        height (int): Height of the grid.
        wind (list of int): Wind strength for each column.
        start_state (tuple): Starting position of the agent.
        goal_state (tuple): Goal position of the agent.
        actions (list of str): Possible actions.
        action_effects (dict): Mapping of actions to state transitions.
    """

    def __init__(self, width, height, wind, start_state, goal_state):

        self.width = width
        self.height = height
        self.wind = wind
        self.start_state = start_state
        self.goal_state = goal_state
        self.actions = ["up", "down", "left", "right"]
        self.action_effects = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }

    def step(self, state, action):
        """
        Take a step in the environment.

        Args:
            state (tuple): Current state (x, y).
            action (str): Action to take.

        Returns:
            next_state (tuple): Next state (x, y).
            reward (int): Reward received.
        """

        x, y = state
        dx, dy = self.action_effects[action]

        # Apply action
        x = min(max(x + dx, 0), self.width - 1)
        y = min(max(y + dy, 0), self.height - 1)

        # Apply wind
        y = min(max(y - self.wind[x], 0), self.height - 1)

        # Check goal
        if (x, y) == self.goal_state:
            return (x, y), 0  # Goal reached, no penalty
        else:
            return (x, y), -1  # Step penalty

    def reset(self):
        """Reset the environment to the start state."""

        return self.start_state
