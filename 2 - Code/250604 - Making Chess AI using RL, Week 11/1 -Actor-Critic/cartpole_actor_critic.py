import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical


# Actor Network
class Actor(nn.Module):
    def __init__(self, input_size, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


# Critic Network
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def save_model(actor, critic, actor_path, critic_path):
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    print(f"Saved actor model to {actor_path}")
    print(f"Saved critic model to {critic_path}")


def load_model(actor, critic, actor_path, critic_path):
    actor.load_state_dict(torch.load(actor_path))
    critic.load_state_dict(torch.load(critic_path))
    actor.eval()
    critic.eval()
    print(f"Loaded actor model from {actor_path}")
    print(f"Loaded critic model from {critic_path}")


def actor_critic(
    actor, critic, episodes, max_steps=2000, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3
):
    optimizer_actor = optim.AdamW(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=lr_critic)
    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}

    env = gym.make("CartPole-v1")
    input_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    for episode in range(1, episodes + 1):
        state = env.reset()[0]
        ep_return = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            state_tensor = torch.FloatTensor(state)

            # Actor selects action
            action_probs = actor(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()

            # Take action and observe next state and reward
            next_state, reward, done, _, _ = env.step(action.item())

            # Critic estimates value function
            value = critic(state_tensor)
            next_value = critic(torch.FloatTensor(next_state))

            # Calculate TD target and Advantage
            td_target = reward + gamma * next_value * (1 - done)
            advantage = td_target - value

            # Critic update with MSE loss
            critic_loss = F.mse_loss(value, td_target.detach())
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # Actor update
            log_prob = dist.log_prob(action)
            actor_loss = -log_prob * advantage.detach()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            # Update state, episode return, and step count
            state = next_state
            ep_return += reward
            step_count += 1

        # Record statistics
        stats["Actor Loss"].append(actor_loss.item())
        stats["Critic Loss"].append(critic_loss.item())
        stats["Returns"].append(ep_return)

        # Print episode statistics
        print(
            f"Episode {episode}: Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Return: {ep_return}, Steps: {step_count}"
        )

    env.close()
    return stats


# Main training loop
if __name__ == "__main__":
    actor_path = "actor_model.pth"
    critic_path = "critic_model.pth"

    # Check if pretrained model exists
    if os.path.exists(actor_path) and os.path.exists(critic_path):
        print("Loading pretrained models...")
        actor = Actor(input_size=4, num_actions=2)
        critic = Critic(input_size=4)
        load_model(actor, critic, actor_path, critic_path)
    else:
        print("Training new models...")
        actor = Actor(input_size=4, num_actions=2)
        critic = Critic(input_size=4)
        episodes = 3000  # Number of episodes for training
        stats = actor_critic(actor, critic, episodes)
        save_model(actor, critic, actor_path, critic_path)

    # Test the trained agent in human mode
    env = gym.make("CartPole-v1", render_mode="human")
    state = env.reset()[0]
    done = False
    total_reward = 0
    max_steps = 2000  # Maximum steps per episode for testing

    while not done:
        env.render()
        state_tensor = torch.FloatTensor(state)
        action_probs = actor(state_tensor)
        action = torch.argmax(action_probs).item()
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if total_reward >= max_steps:
            break

    print(f"Total reward in human mode: {total_reward}")
    env.close()
