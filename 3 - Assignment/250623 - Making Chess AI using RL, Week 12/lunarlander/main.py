import random
import gymnasium as gym
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

# Initialize TensorBoard writer
writer = SummaryWriter()

num_episodes = 600
batch_size = 128
GAMMA = 0.99
LR = 1e-4
TAU = 0.005

EPSILON = 1.0  # Start with full exploration
EPSILON_MIN = 0.01  # Minimum value
EPSILON_DECAY = 0.995  # Decay factor per episode

reward_list = []
episode_durations = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple(
    "Transition", ["state", "action", "next_state", "reward", "done"]
)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        # TODO: Define the neural network architecture
        pass

    def forward(self, x):
        # TODO: Implement the forward pass
        pass


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # TODO: Store a transition in memory
        pass

    def sample(self, batch_size):
        # TODO: Sample a batch of transitions from memory
        pass

    def __len__(self):
        return len(self.memory)


# Initialize the environment
env = gym.make("LunarLander-v3", render_mode="human")

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

replay_memory = ReplayMemory(10000)


def select_action(state):
    # TODO: Implement epsilon-greedy action selection
    pass


optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()

for episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0

    for t in count():
        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        # TODO: Convert next_state and reward to tensors
        # TODO: Store the transition in replay memory

        state = next_state
        total_reward += reward.item()

        if len(replay_memory) >= batch_size:
            # TODO: Sample a batch of transitions from replay memory
            # TODO: Calculate the target Q-value and policy Q-value
            # TODO: Compute the loss and back propagate
            loss = 0
            # TODO: Gradient clipping

            optimizer.step()

            # Log loss to TensorBoard
            writer.add_scalar("Loss", loss.item(), episode)

        # TODO: Update target network
        
        if done:
            episode_durations.append(t + 1)
            reward_list.append(total_reward)

            # Log metrics to TensorBoard
            writer.add_scalar("Reward", total_reward, episode)
            writer.add_scalar("Episode Duration", t + 1, episode)
            writer.add_scalar("Epsilon", EPSILON, episode)
            break

    # TODO: Decay epsilon

# Close TensorBoard writer
writer.close()

# Save the trained model
torch.save(policy_net.state_dict(), "models/dqn_lunar_lander.pth")
print("Model saved successfully!")

print("Complete")
env.close()
