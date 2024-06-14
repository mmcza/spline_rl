import gymnasium as gym
from gymnasium.spaces import flatten
import numpy as np
import os
from datetime import datetime
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation as R
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.world import World
import math
import csv
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Custom Environment for the Wall Hitting Drone Task
class Wall_Hitting_Drone_Env(gym.Env):
    def __init__(self, quad_params, world, trajectory_length=501, render_mode=None):
        super(Wall_Hitting_Drone_Env, self).__init__()
         # Initialize the underlying Quadrotor environment
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
        # Define observation space and action space for the environment
        self.observation_space = gym.spaces.Dict({
            'x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'q': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            'v': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'w': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })

        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(6, 3), dtype=np.float32)

        self.trajectory_length = trajectory_length
        self.trajectory_idx = 0
        self.state = self._generate_initial_state()
        self.reward_given = False

    def reset(self, trajectory=None, seed=None, options=None):
            # Reset environment to initial state
        state = self._generate_initial_state()
        self.trajectory_idx = 0
        self.reward_given = False
        # Ensure the 'q' key is included in the observation
        return {'x': state['x'], 'v': state['v'], 'q': state['q'], 'w': state['w']}, {}

    def step(self, action):
          # Execute one step in the environment
        observation, reward, terminated, truncated, _ = self.env.step(action)

        self.state = {
            'x': observation[:3],
            'v': observation[3:6],
            'q': observation[6:10],
            'w': observation[10:13]
        }

        self.trajectory_idx += 1
         # Check if the trajectory has ended
        if not truncated:
            if self.trajectory_idx >= self.trajectory_length:
                truncated = True
        # Check if the reward has already been given
        if self.reward_given:
            terminated = True

        state_array = np.concatenate([self.state['x'], self.state['v'], self.state['q'], self.state['w']])

        return self.state, reward, terminated, truncated, {}

    def _generate_initial_state(self):
           # Generate the initial state of the environment
        state = self.env.reset()[0]
        return {
            'x': state[:3],
            'v': state[3:6],
            'q': state[6:10],
            'w': state[10:13]
        }

    def _compute_reward(self, state, action):
          # Custom reward function based on the state and action
        if np.abs(state[0] - 5.0) < 0.05 and not self.reward_given:
            self.reward_given = True
            quaternion = state[6:10]
            euler_angles = R.from_quat(quaternion).as_euler('xyz', degrees=False)
            pitch = euler_angles[0]
            print("Success!")
            return 500 - np.exp(state[4]-5.0) * 0.001 - np.exp(abs(pitch)) * 0.05 - np.exp(abs(state[10])) * 0.01
        else: 
            return -np.exp(5.0 - state[0]) * 0.001

# Custom Observation Wrapper to flatten the observation space
class CustomObsWrapper(gym.ObservationWrapper):    
    def __init__(self, env):
        super(CustomObsWrapper, self).__init__(env)
        self.observation_space = gym.spaces.flatten_space(env.observation_space)

    def observation(self, obs):
        # Only include keys that are present in the observation space
        obs = {key: obs[key] for key in self.env.observation_space.spaces.keys() if key in obs}
        return gym.spaces.flatten(self.env.observation_space, obs)
        
# Functions to calculate and save trajectories
def calculate_trajectory_orientation_velocity_acceleration(b_splined_trajectory):
    directions = np.diff(b_splined_trajectory, axis=0)
    distances = np.linalg.norm(directions, axis=1)
    unit_directions = directions / distances[:, None]

    rotations = R.from_rotvec(unit_directions)
    # Calculate linear speeds and accelerations
    linear_speeds = np.diff(b_splined_trajectory, axis=0) / 0.01
    linear_speeds = np.pad(linear_speeds, ((0, 1), (0, 0)), 'edge')
    linear_accelerations = np.diff(linear_speeds, axis=0) / 0.01
    linear_accelerations = np.pad(linear_accelerations, ((0, 1), (0, 0)), 'edge')
    # Calculate yaw angles and their rates and accelerations
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

def save_models(model, episode, save_dir='models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(os.path.join(save_dir, f'ppo_model_{episode}'))

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
    # Load quadrotor parameters and world map
    quadrotor_params = quad_params
    world_map = {"bounds": {"extents": [-10., 10., -10., 10., -10, 10.]}}
    world = World(world_map)
    # Initialize the custom environment
    env = Wall_Hitting_Drone_Env(quadrotor_params, world, 501, render_mode=None)
    env = CustomObsWrapper(env)
    env = DummyVecEnv([lambda: env])

    # Create the PPO model
    model = PPO(MlpPolicy, env, verbose=1, ent_coef=0.01, tensorboard_log="./ppo_quadrotor_tensorboard/")

    num_episodes = 100000 
    rewards_to_save = []
    # Initialize the controller
    controller = SE3Control(quadrotor_params)
      # Get the current time for saving files
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    #Training loop
    for episode in range(num_episodes):
        obs = env.envs[0].reset()
        state = env.envs[0].state
        total_reward = 0
        done = False
          # Flatten the state for the observation space
        state_array = np.concatenate([state['x'], state['v'], state['q'], state['w']])

        obs = flatten(env.envs[0].observation_space, state_array)

        # Generate trajectory using the trained model
        trajectory, _ = model.predict(obs, deterministic=True)
        traj = np.copy(trajectory)
        #print(state)
        traj = np.insert(traj, 0, state['x'], axis=0)
         # Create a smooth spline trajectory
        t = np.linspace(0, 5, 7)
        t_new = np.linspace(0, 5, 501)
        b_splined_trajectory = [make_interp_spline(t, traj[:, i], k=3)(t_new) for i in range(trajectory.shape[1])]
        b_splined_trajectory = np.array(b_splined_trajectory).T
           # Calculate trajectory parameters
        yaw_angles, linear_speeds, linear_accelerations, yaw_rates, yaw_accelerations = calculate_trajectory_orientation_velocity_acceleration(b_splined_trajectory)
        
        while not done:
               # Get the next point in the trajectory
            next_point = trajectory_to_dict(b_splined_trajectory, yaw_angles, linear_speeds, linear_accelerations, yaw_rates, yaw_accelerations, env.envs[0].trajectory_idx)
            action = controller.update(env.envs[0].trajectory_idx * env.envs[0].Tp, state, next_point) #update the controller and get the action
            next_state, reward, truncated, terminated, info = env.envs[0].step(action['cmd_motor_speeds'])
            #update the state
            state = {
                'x': next_state[:3],
                'v': next_state[3:6],
                'q': next_state[6:10],
                'w': next_state[10:13]
            }
            total_reward += reward
                # Check if the episode has ended
            if truncated or terminated:
                done = True
        
        model.learn(total_timesteps=2048)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        rewards_to_save.append([episode + 1, total_reward])
         # Save models and trajectories every 100 episodes
        if (episode + 1) % 100 == 0:
            save_models(model, episode + 1, f"ppo_rl_{current_time}")
            save_trajectory(b_splined_trajectory, episode + 1, f"ppo_rl_{current_time}/trajectories.csv")
            save_rewards(rewards_to_save, f"ppo_rl_{current_time}/rewards.csv")
            mean_reward = np.mean([reward for _, reward in rewards_to_save])
            if mean_reward > 250:
                break
            rewards_to_save = []

    if rewards_to_save:
        save_models(model, episode + 1)
        save_trajectory(b_splined_trajectory, episode + 1, 'trajectories.csv')
        save_rewards(rewards_to_save, 'rewards.csv')

    print("Training complete.")
