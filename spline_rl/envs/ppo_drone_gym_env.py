import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation as R
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.world import World
import math
import os
import csv

class Wall_Hitting_Drone_Env(gym.Env):
    def __init__(self, quad_params, world, trajectory_length=501, render_mode=None):
        super(Wall_Hitting_Drone_Env, self).__init__()

        # Create the Quadrotor-v0 environment
        self.env = gym.make("Quadrotor-v0",
                            control_mode='cmd_motor_speeds',
                            reward_fn=self._compute_reward,
                            quad_params=quad_params,
                            max_time=5,
                            world=world,
                            sim_rate=100,
                            render_mode=render_mode,
                            render_fps=30)

        self.Tp = 0.01

        # Define the observation space (state of the drone)
        self.observation_space = gym.spaces.Dict({
            'x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'q': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            'v': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'w': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })

        # Define the action space (trajectory control points)
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(11, 6), dtype=np.float32)

        self.trajectory_length = trajectory_length
        self.trajectory_idx = 0
        self.state = self._generate_initial_state()
        self.reward_given = False

    def reset(self, trajectory=None):
        self.state = self._generate_initial_state()
        self.trajectory_idx = 0
        self.reward_given = False
        return self.state

    def step(self, action):        
        # Execute the action in the environment
        observation, reward, terminated, truncated, _ = self.env.step(action)

        # Update the internal state representation
        self.state = {
            'x': observation[:3],
            'v': observation[3:6],
            'q': observation[6:10],
            'w': observation[10:13]
        }

        self.trajectory_idx += 1

        if not truncated:
            if self.trajectory_idx >= self.trajectory_length:
                truncated = True

        if self.reward_given:
            terminated = True

        return self.state, reward, terminated, truncated

    def _generate_initial_state(self):
        state = self.env.reset()
        return {
            'x': state[0][:3],
            'v': state[0][3:6],
            'q': state[0][6:10],
            'w': state[0][10:13]
        }

    def _compute_reward(self, state, action):
        # Reward function
        if np.abs(state[0] - 5.0) < 0.05 and not self.reward_given:
            self.reward_given = True
            quaternion = state[6:10]
            euler_angles = R.from_quat(quaternion).as_euler('xyz', degrees=False)
            pitch = euler_angles[0]
            return 500 - np.exp(state[4]-5.0) * 0.001 - np.exp(abs(pitch)) * 0.05 - np.exp(abs(state[10])) * 0.01
        else: 
            return -np.exp(5.0 - state[0]) * 0.001


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        return mean


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value.view(-1, 1)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, eps_clip=0.2, k_epochs=10):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value = ValueNetwork(state_dim)
        
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

    def select_action(self, state):
        state = torch.tensor(self._flatten_state(state).clone().detach(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean = self.policy_old(state)
            log_std = self.policy_old.log_std
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).numpy().reshape(11, 3), action_logprob.item()

    def train(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)  # Ensure (batch_size, 1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        old_states = torch.tensor(np.array(memory.states), dtype=torch.float32)
        old_actions = torch.tensor(np.array(memory.actions), dtype=torch.float32)
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32)
        
        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss_fn(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer_policy.zero_grad()
            self.optimizer_value.zero_grad()
            loss.mean().backward()
            self.optimizer_policy.step()
            self.optimizer_value.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

    def evaluate(self, state, action):
        state_value = self.value(state)
        
        mean = self.policy(state)
        log_std = self.policy.log_std
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        return action_logprobs, state_value, dist_entropy
    
    def _flatten_state(self, state):
        flat_state = np.concatenate([
            state['x'],
            state['q'],
            state['v'],
            state['w'],
        ])
        return torch.tensor(flat_state, dtype=torch.float32)

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def calculate_trajectory_orientation_velocity_acceleration(b_splined_trajectory):
    directions = np.diff(b_splined_trajectory, axis=0)
    distances = np.linalg.norm(directions, axis=1)
    unit_directions = directions / distances[:, None]

    rotations = R.from_rotvec(unit_directions)

    linear_speeds = np.diff(b_splined_trajectory, axis=0) / 0.01
    linear_speeds = np.pad(linear_speeds, ((0, 1), (0, 0)), 'edge')
    linear_accelerations = np.diff(linear_speeds, axis=0) / 0.01
    linear_accelerations = np.pad(linear_accelerations, ((0, 1), (0, 0)), 'edge')

    euler_angles = rotations.as_euler('xyz', degrees=False)
    yaw_angles = euler_angles[:, 2]
    yaw_angles = np.pad(yaw_angles, (0, 1), 'edge')

    yaw_rates = np.diff(yaw_angles) / 0.01
    yaw_rates = np.pad(yaw_rates, (0, 1), 'edge')
    yaw_accelerations = np.diff(yaw_rates) / 0.01
    yaw_accelerations = np.pad(yaw_accelerations, (0, 1), 'edge')

    return yaw_angles, linear_speeds, linear_accelerations, yaw_rates, yaw_accelerations

def trajectory_to_dict(trajectory, yaw_angles, linear_speeds, linear_accelerations, angular_speeds, angular_accelerations, index):
    trajectory_dict = {
        'x': trajectory[index],
        'yaw': yaw_angles[index],
        'x_dot': linear_speeds[index],
        'x_ddot': linear_accelerations[index],
        'x_dddot': np.zeros(3),
        'x_ddddot': np.zeros(3),
        'yaw_dot': angular_speeds[index],
        'yaw_ddot': angular_accelerations[index]
    }
    return trajectory_dict


def save_models(policy, value, episode, save_dir='models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(policy.state_dict(), os.path.join(save_dir, f'policy_model_{episode}.pth'))
    torch.save(value.state_dict(), os.path.join(save_dir, f'value_model_{episode}.pth'))

def save_trajectory(trajectory, episode, file_path='trajectories.csv'):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for point in trajectory:
            writer.writerow([episode, *point])

def save_rewards(rewards, file_path='rewards.csv'):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rewards)

def initialize_trajectory_file(file_path='trajectories.csv'):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'X', 'Y', 'Z']) 

# Main Training Loop
if __name__ == "__main__":
    quadrotor_params = quad_params
    world_map = {"bounds": {"extents": [-10., 10., -10., 10., -10, 10.]}}
    world = World(world_map)

    state_dim = 13  # Dimension of state representation
    action_dim = 33  # 11 points * 3 dimensions for trajectory

    env = Wall_Hitting_Drone_Env(quadrotor_params, world, 501, render_mode=None)
    agent = PPOAgent(state_dim=state_dim, action_dim=33)  # action_dim should be 11 * 3 = 33

    memory = Memory()
    num_episodes = 100000
    rewards_to_save = []

    controller = SE3Control(quadrotor_params)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        # Generate the trajectory using the policy
        trajectory, action_logprob = agent.select_action(state)
        
        # Create b-spline from trajectory
        t = np.linspace(0, 5, 11)
        t_new = np.linspace(0, 5, 501)
        b_splined_trajectory = [make_interp_spline(t, trajectory[:, i], k=3)(t_new) for i in range(trajectory.shape[1])]
        b_splined_trajectory = np.array(b_splined_trajectory).T

        # Calculate orientation, velocity, and acceleration of the trajectory
        yaw_angles, linear_speeds, linear_accelerations, yaw_rates, yaw_accelerations = calculate_trajectory_orientation_velocity_acceleration(b_splined_trajectory)
        
        while not done:
            next_point = trajectory_to_dict(b_splined_trajectory, yaw_angles, linear_speeds, linear_accelerations, yaw_rates, yaw_accelerations, env.trajectory_idx)
            action = controller.update(env.trajectory_idx * env.Tp, state, next_point)
            next_state, reward, truncated, terminated = env.step(action['cmd_motor_speeds'])
            
            memory.states.append(agent._flatten_state(state).numpy())
            memory.actions.append(trajectory.flatten())
            memory.logprobs.append(action_logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(truncated or terminated)
            
            state = next_state
            total_reward += reward
            
            if truncated or terminated:
                done = True
        
        agent.train(memory)
        memory.clear_memory()
        
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        rewards_to_save.append([episode + 1, total_reward])

        # Save models, trajectory and rewards every 100 episodes
        if (episode + 1) % 100 == 0:
            save_models(agent.policy, agent.value, episode + 1)
            save_trajectory(b_splined_trajectory, episode + 1, 'trajectories.csv')
            save_rewards(rewards_to_save, 'rewards.csv')
            mean_reward = np.mean([reward for _, reward in rewards_to_save])
            if mean_reward > 250:
                break
            rewards_to_save = []

    if rewards_to_save:
        save_models(agent.policy, agent.value, episode + 1)
        save_trajectory(b_splined_trajectory, episode + 1, 'trajectories.csv')
        save_rewards(rewards_to_save, 'rewards.csv')

    print("Training complete.") 