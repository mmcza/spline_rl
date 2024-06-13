import os
import torch
import numpy as np
import torch.nn as nn
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.world import World
from spline_rl.envs.ppo_drone_gym_env import Wall_Hitting_Drone_Env, calculate_trajectory_orientation_velocity_acceleration, trajectory_to_dict
from spline_rl.envs.ppo_drone_gym_env import PolicyNetwork, ValueNetwork
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation as R

def get_latest_model(model_dir='models', model_type='policy'):
    model_files = [f for f in os.listdir(model_dir) if f.startswith(model_type) and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"No {model_type} model files found in the directory.")
    
    latest_model_file = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(model_dir, latest_model_file)

def load_models(policy, value, policy_model_file, value_model_file):
    policy.load_state_dict(torch.load(policy_model_file))
    value.load_state_dict(torch.load(value_model_file))
    policy.float()  # Ensure the policy model is in float32
    value.float()   # Ensure the value model is in float32

def flatten_state(state):
    flat_state = np.concatenate([
        state['x'],
        state['q'],
        state['v'],
        state['w'],
    ])
    return torch.tensor(flat_state, dtype=torch.float32)  # Ensure float32 dtype

def main():
    quadrotor_params = quad_params
    world_map = {"bounds": {"extents": [-10., 10., -10., 10., -10, 10.]}}
    world = World(world_map)

    # Load the environment
    env = Wall_Hitting_Drone_Env(quadrotor_params, world, 501, render_mode='3D')

    state_dim = 13  # Dimension of state representation
    action_dim = 33  # 11 points * 3 dimensions for trajectory

    # Initialize the policy and value networks
    policy = PolicyNetwork(state_dim, action_dim)
    value = ValueNetwork(state_dim)

    # Get the latest models
    latest_policy_model_file = get_latest_model(model_type='policy')
    latest_value_model_file = get_latest_model(model_type='value')
    load_models(policy, value, latest_policy_model_file, latest_value_model_file)

    # Set the policy to evaluation mode
    policy.eval()

    # Experiment with the loaded model
    state = env.reset()
    total_reward = 0
    done = False
    controller = SE3Control(quadrotor_params)
    
    with torch.no_grad():
        while not done:
            state_tensor = flatten_state(state).unsqueeze(0)
            trajectory = policy(state_tensor).squeeze(0).numpy().reshape(11, 3)
            
            # Create b-spline from trajectory
            t = np.linspace(0, 5, 11)
            t_new = np.linspace(0, 5, 501)
            b_splined_trajectory = [make_interp_spline(t, trajectory[:, i], k=3)(t_new) for i in range(trajectory.shape[1])]
            b_splined_trajectory = np.array(b_splined_trajectory).T
            
            # Calculate orientation, velocity, and acceleration of the trajectory
            yaw_angles, linear_speeds, linear_accelerations, yaw_rates, yaw_accelerations = calculate_trajectory_orientation_velocity_acceleration(b_splined_trajectory)
            
            next_point = trajectory_to_dict(b_splined_trajectory, yaw_angles, linear_speeds, linear_accelerations, yaw_rates, yaw_accelerations, env.trajectory_idx)
            action = controller.update(env.trajectory_idx * env.Tp, state, next_point)
            next_state, reward, truncated, terminated = env.step(action['cmd_motor_speeds'])
            
            state = next_state
            total_reward += reward
            
            if truncated or terminated:
                done = True
    
    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    main()