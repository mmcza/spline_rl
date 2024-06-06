import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from rotorpy.world import World

from rotorpy.vehicles.crazyflie_params import quad_params  

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv

# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import hover_reward
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch

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
               control_mode='cmd_motor_speeds',
               quad_params=quad_params,
               max_time=5,
               world=world,
               sim_rate=100,
               render_mode='3D')

# Reset the environment
observation, info = env.reset(initial_state='random', options={'pos_bound': 2, 'vel_bound': 0})

# Create a new model
model = PPO(MlpPolicy, env, verbose=1, ent_coef=0.01, tensorboard_log=tensorboard_log_dir)

# Training parameters
num_timesteps = 20_000
num_epochs = 10
start_time = datetime.now()
epoch_count = 0

# Evaluation callback
eval_env = gym.make("Quadrotor-v0",
                    control_mode='cmd_motor_speeds',
                    quad_params=quad_params,
                    max_time=5,
                    world=world,
                    sim_rate=100,
                    render_mode='None')

eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                             log_path=log_dir, eval_freq=5000,
                             deterministic=True, render=False)

# Training loop with visualization and monitoring
try:
    while epoch_count < num_epochs:  
        # Train the model
        model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False,
                    tb_log_name="PPO-Quad_cmd-motor_" + start_time.strftime('%H-%M-%S'),
                    callback=eval_callback)

        # Save the model
        model_path = f"{models_dir}/PPO/{start_time.strftime('%H-%M-%S')}/hover_{num_timesteps*(epoch_count+1)}"
        model.save(model_path)

        # Evaluate the model
        obs = eval_env.reset()
        done = False
        rewards = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            rewards.append(reward)

        epoch_count += 1

        # Plotting results
        plt.plot(range(len(rewards)), rewards)
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title(f'Rewards in Epoch {epoch_count}')
        plt.show()
        print(f"Epoch {epoch_count}/{num_epochs} completed, model saved to {model_path}")

except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    model.save(f"{models_dir}/PPO/interrupt_{start_time.strftime('%H-%M-%S')}")

plt.plot(range(epoch_count), rewards)
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('Training Reward Over Time')
plt.show()
