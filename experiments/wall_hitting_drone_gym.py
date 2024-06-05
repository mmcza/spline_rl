import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from rotorpy.world import World

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv

# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import hover_reward

"""
In this script, we demonstrate how to train a hovering control policy in RotorPy using Proximal Policy Optimization. 
We use our custom quadrotor environment for Gymnasium along with stable baselines for the PPO implementation. 

The task is for the quadrotor to stabilize to hover at the origin when starting at a random position nearby. 

Training can be tracked using tensorboard, e.g. tensorboard --logdir=<log_dir>

"""

# First we'll set up some directories for saving the policy and logs.
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "logs")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Next import Stable Baselines.
try:
    import stable_baselines3
except:
    raise ImportError('To run this example you must have Stable Baselines installed via pip install stable_baselines3')

from stable_baselines3 import PPO                                   # We'll use PPO for training.
from stable_baselines3.ppo.policies import MlpPolicy                # The policy will be represented by an MLP

num_cpu = 4   # for parallelization

world_map = {"bounds": {"extents": [-10., 10., -10., 10., -0.5, 10.]},
        "blocks": [{"extents": [-5, -5.5, -10., 10., -0.5, 10.], "color": [1, 0, 0]}]}
        
world = World(world_map)

# Make the environment. For this demo we'll train a policy to command collective thrust and body rates.
# Turning render_mode="None" will make the training run much faster, as visualization is a current bottleneck. 
env = gym.make("Quadrotor-v0", 
                control_mode ='cmd_motor_speeds', 
                quad_params = quad_params,
                max_time = 5,
                world = world,
                sim_rate = 100,
                #render_mode='None')
                render_mode='3D')

# from stable_baselines3.common.env_checker import check_env
# check_env(env, warn=True)  # you can check the environment using built-in tools

# Reset the environment
observation, info = env.reset(initial_state='random', options={'pos_bound': 2, 'vel_bound': 0})

# Create a new model
model = PPO(MlpPolicy, env, verbose=1, ent_coef=0.01, tensorboard_log=log_dir)

# Training... 
num_timesteps = 20_000
num_epochs = 10

start_time = datetime.now()

epoch_count = 0
while True:  # Run indefinitely..

    # This line will run num_timesteps for training and log the results every so often.
    model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False, tb_log_name="PPO-Quad_cmd-motor_"+start_time.strftime('%H-%M-%S'))

    # Save the model
    model.save(f"{models_dir}/PPO/{start_time.strftime('%H-%M-%S')}/hover_{num_timesteps*(epoch_count+1)}")

    epoch_count += 1