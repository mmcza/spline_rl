import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime
from rotorpy.world import World
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.learning.quadrotor_environments import QuadrotorEnv

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.relu(self.l1(torch.cat([x, u], 1)))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)

# Define the DDPG agent
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.replay_buffer = ReplayBuffer()
        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=64, discount=0.99, tau=0.005):
        if self.replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)
        q_targets = rewards + (1 - dones) * discount * next_q_values
        q_values = self.critic(states, actions)
        q_loss = nn.MSELoss()(q_values, q_targets.detach())

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Setting up directories for saving policies and logs
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "..", "rotorpy", "learning", "policies")
log_dir = os.path.join(current_dir, "..", "rotorpy", "learning", "logs")
tensorboard_log_dir = os.path.join(log_dir, "tensorboard")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

# Set up the environment
world_map = {
    "bounds": {"extents": [-10., 10., -10., 10., -0.5, 10.]},
    "blocks": [{"extents": [-5, -5.5, -0.5, 5.5, -0.5, 10.], "color": [1, 0, 0]}]
}
world = World(world_map)

def make_env(render_mode=None):
    class CustomQuadrotorEnv(QuadrotorEnv):
        def step(self, action):
            obs, reward, done, truncated, info = super().step(action)
            block_x_position = -5
            block_y_position = -5.5
            block_z_position= -0.5

            drone_x_position = obs[0]
            drone_y_position = obs[1]
            drone_z_position = obs[2]

            distance_to_block = abs(block_x_position - drone_x_position) + \
                                abs(block_y_position - drone_y_position) + \
                                abs(block_z_position - drone_z_position)
            reward = -distance_to_block
            reward -= 0.01

            if distance_to_block < 2.0:
                done = True
                reward += 100

            return obs, reward, done, truncated, info

    env = CustomQuadrotorEnv(
        control_mode='cmd_motor_speeds',
        quad_params=quad_params,
        max_time=2,
        world=world,
        sim_rate=100,
        render_mode=render_mode
    )
    return env

env = make_env(render_mode=None)
eval_env = make_env(render_mode='3D')

# Define hyperparameters
num_episodes = 1500  # Reduced number of episodes
batch_size = 64
discount = 0.99
tau = 0.005
max_action = 1.0
start_timesteps = 1000  # Reduced initial random action steps
expl_noise = 0.1  # Reduced exploration noise
eval_freq = 50  # Evaluate more frequently to monitor progress

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DDPG(state_dim, action_dim, max_action)

# Training loop
total_timesteps = 0
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    while not done:
        total_timesteps += 1
        if total_timesteps < start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.array(state))
            action = action + np.random.normal(0, expl_noise, size=action_dim)
            action = action.clip(env.action_space.low, env.action_space.high)

        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        agent.replay_buffer.add((state, action, reward, next_state, float(done)))
        state = next_state
        episode_reward += reward

        if total_timesteps >= start_timesteps:
            agent.train(batch_size, discount, tau)

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}, Reward: {episode_reward}, Total Timesteps: {total_timesteps}")

    # Save the model and evaluate
    if (episode + 1) % eval_freq == 0:
        model_path = os.path.join(models_dir, f"DDPG_{episode + 1}")
        os.makedirs(model_path, exist_ok=True)
        torch.save(agent.actor.state_dict(), os.path.join(model_path, "actor.pth"))
        torch.save(agent.critic.state_dict(), os.path.join(model_path, "critic.pth"))

        # Evaluate the model
        eval_reward = 0
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(np.array(state))
            state, reward, done, truncated, _ = eval_env.step(action)
            done = done or truncated
            eval_reward += reward

        print(f"Evaluation Reward after {episode + 1} episodes: {eval_reward}")

plt.figure()
plt.plot(range(len(episode_rewards)), episode_rewards)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Training Reward Over Episodes')
plt.show()

model_path = os.path.join(models_dir, "DDPG_final")
os.makedirs(model_path, exist_ok=True)
torch.save(agent.actor.state_dict(), os.path.join(model_path, "actor.pth"))
torch.save(agent.critic.state_dict(), os.path.join(model_path, "critic.pth"))
