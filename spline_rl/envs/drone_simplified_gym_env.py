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
            return 100
        else: 
            return -np.exp(5.0 - state[0]) * 0.001


class TrajectoryGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TrajectoryGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 11, 3)

class Agent:
    def __init__(self, state_dim, action_dim):
        self.model = TrajectoryGenerator(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def generate_trajectory(self, state):
        state = torch.tensor(self._flatten_state(state), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            control_points = self.model(state)
        return control_points.squeeze(0).numpy()

    def train(self, state, action, reward):
        state = torch.tensor(self._flatten_state(state), dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)

        self.optimizer.zero_grad()
        predicted_action = self.model(state)
        #loss = self.loss_fn(predicted_action, action)
        loss = -torch.sum(reward)
        # loss.backward()
        self.optimizer.step()

    def _flatten_state(self, state):
        flat_state = np.concatenate([
            state['x'],
            state['q'],
            state['v'],
            state['w'],
        ])
        return torch.tensor(flat_state, dtype=torch.float32)

state_dim = 13  # Dimension of state representation
action_dim = 33  # 11 points * 3 dimensions for trajectory

# Calculate trajectory orientation, velocity and acceleration
def calculate_trajectory_orientation_velocity_acceleration(b_splined_trajectory):
    # Calculate direction vectors
    directions = np.diff(b_splined_trajectory, axis=0)
    distances = np.linalg.norm(directions, axis=1)
    unit_directions = directions / distances[:, None]

    # Calculate rotations
    rotations = R.from_rotvec(unit_directions)

    # Calculate linear speed and acceleration for each axis separately
    linear_speeds = np.diff(b_splined_trajectory, axis=0) / 0.01
    linear_speeds = np.pad(linear_speeds, ((0, 1), (0, 0)), 'edge')
    linear_accelerations = np.diff(linear_speeds, axis=0) / 0.01
    linear_accelerations = np.pad(linear_accelerations, ((0, 1), (0, 0)), 'edge')

    # Convert rotation vector to Euler angles (roll, pitch, yaw)
    euler_angles = rotations.as_euler('xyz', degrees=False)

    # Extract yaw angles
    yaw_angles = euler_angles[:, 2]
    yaw_angles = np.pad(yaw_angles, (0, 1), 'edge')

    # Calculate yaw rates and accelerations
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

# Main Training Loop
if __name__ == "__main__":
    quadrotor_params = quad_params
    world_map = {"bounds": {"extents": [-10., 10., -10., 10., -10, 10.]}}
    world = World(world_map)

    env = Wall_Hitting_Drone_Env(quadrotor_params, world, 501, render_mode=None) # render_mode = '3D' to visualize the environment
    agent = Agent(state_dim=state_dim, action_dim=action_dim)

    num_episodes = 1000
    gamma = 0.99  # Discount factor

    controller = SE3Control(quadrotor_params)

    t = np.linspace(0, 5, 11)
    t_new = np.linspace(0, 5, 501)

    for episode in range(num_episodes):
        # Reset the environment and generate a trajectory
        initial_state = env.reset()
        trajectory = agent.generate_trajectory(initial_state)

        # Create b_spline from trajectory
        b_splined_trajectory = [make_interp_spline(t, trajectory[:, i], k=5)(t_new) for i in range(trajectory.shape[1])]
        b_splined_trajectory = np.array(b_splined_trajectory).T

        # Calculate orientation, velocity, and acceleration of the trajectory
        yaw_angles, linear_speeds, linear_accelerations, yaw_rates, yaw_accelerations = calculate_trajectory_orientation_velocity_acceleration(b_splined_trajectory)

        # Initialize the state, done flag, and total reward        
        state = initial_state
        done = False
        total_reward = 0

        while not done:
            next_point = trajectory_to_dict(b_splined_trajectory, yaw_angles, linear_speeds, linear_accelerations, yaw_rates, yaw_accelerations, env.trajectory_idx)
            action = controller.update(env.trajectory_idx * env.Tp, state, next_point)
            # print(action)
            next_state, reward, truncated, terminated = env.step(action['cmd_motor_speeds'])
            if truncated or terminated:
                done = True
            total_reward += reward
            state = next_state

        agent.train(initial_state, trajectory, total_reward)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    print("Training complete.")

