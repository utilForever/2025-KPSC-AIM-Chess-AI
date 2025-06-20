import random
import time

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer

env_id = "BreakoutNoFrameskip-v4"
num_envs = 1
total_timesteps = 5_000_000
learning_rate = 1e-4
buffer_size = 250_000
gamma = 0.99
tau = 1.0
target_network_frequency = 1000
batch_size = 32
start_eps = 1.0
end_eps = 0.01
exploration_duration = 1_000_000
num_steps_before_training = 80_000
train_frequency = 4
seed = None
video_path = "videos_dqn_atari"


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


class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, envs.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_eps: float, end_eps: float, duration: int, t: int):
    # TODO: Implement linear schedule for epsilon-greedy exploration
    raise NotImplementedError


if seed is not None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


# Make environments
envs = gym.vector.SyncVectorEnv(
    [
        lambda: make_env(env_id, False, None if seed is None else seed + i)
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

# Make neural networks
q_network = QNetwork(envs).to(device)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
target_network = QNetwork(envs).to(device)
target_network.load_state_dict(q_network.state_dict())

rb = ReplayBuffer(
    buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    optimize_memory_usage=True,
    handle_timeout_termination=False,
)


def np2torch(a, dtype=torch.float32, device=device):
    return torch.as_tensor(a, dtype=dtype, device=device)


# Training loop

episode_returns = []
losses = []

start_time = time.time()
obs, _ = envs.reset(seed=seed)

for global_step in range(total_timesteps):

    # TODO: Select an action using epsilon-greedy policy
    raise NotImplementedError

    # Take a step in the environment
    next_obs, rewards, dones, truncateds, infos = envs.step(actions)

    for i, (done, truncated) in enumerate(zip(dones, truncateds)):
        if done or truncated:
            # This only appears when all lives are lost
            if "episode" in infos["final_info"]:
                ret = infos["final_info"]["episode"]["r"][0]
                episode_returns.append(ret)
                print(
                    f"global_step={global_step}, episode={len(episode_returns)}, episode_return={ret}"
                )

    # Since we use same step
    real_next_obs = next_obs.copy()
    for i, truncated in enumerate(truncateds):
        if truncated:
            real_next_obs[i] = infos["final_obs"][i]
    rb.add(obs, real_next_obs, actions, rewards, dones, infos)
    obs = next_obs

    # Training
    if global_step > num_steps_before_training:
        if global_step % train_frequency == 0:
            data = rb.sample(batch_size)

            # TODO: Calculate TD targets and compute loss
            raise NotImplementedError

            # TODO: Update the optimizer
            raise NotImplementedError

            if global_step % 100 == 0:
                print(
                    "Steps per second:", int(global_step / (time.time() - start_time))
                )

            # Update target network
            if global_step % target_network_frequency == 0:
                # TODO: Soft update using parameter tau
                raise NotImplementedError

# Close env
envs.close()

model_path = f"dqn_atari_{env_id}.pth"
print(f"Saving model to {model_path}")
torch.save(q_network.state_dict(), model_path)
