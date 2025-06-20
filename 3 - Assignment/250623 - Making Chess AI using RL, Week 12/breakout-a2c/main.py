import random
import time

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

env_id = "BreakoutNoFrameskip-v4"
num_envs = 16
n_steps = 5
total_timesteps = 10_000_000
learning_rate = 2e-4
gamma = 0.99
gae_lambda = 0.95
entropy_weight = 0.01
value_weight = 0.25
max_norm = 0.5
seed = None
video_path = "videos_a2c_gae_atari"


def make_env(env_id, capture_video, seed=None):
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, video_path, episode_trigger=lambda episode_id: True
        )
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)

    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)

    if seed is not None:
        env.action_space.seed(seed)

    return env


class ActorCritic(nn.Module):
    def __init__(self, envs):
        super().__init__()

        n_input_channels = envs.single_observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(envs.single_observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
        )

        # Actor (Policy) Head
        self.actor = nn.Linear(512, envs.single_action_space.n)
        # Critic (Value Function) Head
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        features = self.cnn(x / 225.0)
        shared_latent = self.linear(features)

        return self.actor(shared_latent), self.critic(shared_latent)


# Make environments
envs = gym.vector.SyncVectorEnv(
    env_fns=[
        lambda: make_env(env_id, False, seed + i if seed is not None else None)
        for i in range(num_envs)
    ],
    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
)

# Set device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Make neural network
ac_network = ActorCritic(envs).to(device)
optimizer = optim.Adam(ac_network.parameters(), lr=learning_rate, eps=1e-5)

rollout_buffer = RolloutBuffer(
    buffer_size=n_steps,
    observation_space=envs.single_observation_space,
    action_space=envs.single_action_space,
    device=device,
    gamma=gamma,
    gae_lambda=gae_lambda,
    n_envs=num_envs,
)


def sample_action(logits):
    # TODO: Sample an action from logits
    raise NotImplementedError


def compute_entropy_and_log_probs(policy_logits, actions):
    # TODO: Compute entropy for loss regularization
    #       and return the mean of the entropy and log probabilities
    raise NotImplementedError


def np2torch(a, dtype=torch.float32, device=device):
    return torch.as_tensor(a, dtype=dtype, device=device)


# Training loop
episode_returns = []
losses = []

global_step = 0
start_time = time.time()

# Reset env
obs, _ = envs.reset(seed=seed)

# Main training loop
while global_step < total_timesteps:

    # Rollout phase
    for step in range(n_steps):
        # Convert obs to tensor
        obs_tensor = np2torch(obs)

        # Forward pass through the network
        with torch.no_grad():
            policy_logits, values = ac_network(obs_tensor)
            actions = sample_action(policy_logits)
            _, log_probs = compute_entropy_and_log_probs(policy_logits, actions)

        # Take actions in the environment
        actions_np = actions.cpu().numpy()
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions_np)
        dones = terminateds | truncateds

        # Store data in buffer
        rollout_buffer.add(
            obs,
            actions_np,
            rewards,
            terminateds,
            values.squeeze(-1),
            log_probs,
        )

        # Update observations
        obs = next_obs

        for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
            if terminated or truncated:
                if "episode" in infos["final_info"]:
                    ret = infos["final_info"]["episode"]["r"][i]
                    episode_returns.append(ret)
                    print(
                        f"global_step={global_step}, episode={len(episode_returns)}, episode_return={ret}"
                    )

    # Update phase
    with torch.no_grad():
        _, last_values = ac_network(np2torch(next_obs))
    last_dones = dones

    # TODO: Calculate advantages and returns in the rollout buffer
    # HINT: δ_t = r_t + γ V_{t+1}(1-done) - V_t
    #       A_t = δ_t + γλ(1-done) A_{t+1}
    #       R_t = A_t + V_t

    # This will only loop once (get all data in one go)
    for rollout_data in rollout_buffer.get(batch_size=None):

        observations_arr = rollout_data.observations
        actions_arr = rollout_data.actions.flatten()
        advantages_arr = rollout_data.advantages
        returns_arr = rollout_data.returns

        # Calculate
        new_logits, new_values = ac_network(observations_arr)
        entropy, new_log_probs = compute_entropy_and_log_probs(new_logits, actions_arr)
        new_values = new_values.flatten()

        # TODO: Calculate losses: policy loss, value loss, entropy loss and total loss
        raise NotImplementedError

        # Store losses for plotting
        losses.append(loss.item())

        # TODO: Update the optimizer
        raise NotImplementedError

    # Important: reset the buffer
    # If you don't do this, observations will have the wrong shape!
    # It flattens from (N, num_envs, 4, 84, 84) --> (N * num_envs, 4, 84, 84)
    rollout_buffer.reset()

    # Update global step counter
    global_step += n_steps * num_envs

    # Logging
    if global_step % (100 * n_steps * num_envs) == 0 and len(episode_returns) > 100:
        print(f"Global Step: {global_step} / {total_timesteps}")
        print(
            f"  Loss: {loss.item():.4f} (Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f}, Entropy: {-entropy_loss.item():.4f})"
        )

        if episode_returns:
            print(f"  Mean Return (last 100): {np.mean(episode_returns[-100:]):.2f}")

        print(f"  Steps per second: {int(global_step / (time.time() - start_time))}")

# Close env
envs.close()

model_path = f"a2c_gae_atari_{env_id}.pth"
print(f"Saving model to {model_path}")
torch.save(ac_network.state_dict(), model_path)
