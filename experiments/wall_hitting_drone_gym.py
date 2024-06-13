import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from rotorpy.world import World
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_reward_functions import hover_reward
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

"""
In this script, we demonstrate how to train a hovering control policy in RotorPy using Proximal Policy Optimization. 
We use our custom quadrotor environment for Gymnasium along with stable baselines for the PPO implementation. 

The task is for the quadrotor to stabilize to hover at the origin when starting at a random position nearby. 

Training can be tracked using tensorboard, e.g. tensorboard --logdir=<log_dir>

"""

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "..", "rotorpy", "learning", "policies")
log_dir = os.path.join(current_dir, "..", "rotorpy", "learning", "logs")
tensorboard_log_dir = os.path.join(log_dir, "tensorboard")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

# Set world map for the environment
world_map = {
    "bounds": {"extents": [-10., 10., -10., 10., -0.5, 10.]},
    "blocks": [{"extents": [-5, -5.5, -10., 10., -0.5, 10.], "color": [1, 0, 0]}]
}
world = World(world_map)

# Initialize the environment
env = gym.make("Quadrotor-v0",
               control_mode='cmd_motor_speeds',
               quad_params=quad_params,
               max_time=5,
               world=world,
               sim_rate=100,
               render_mode='3D')

# Reset the environment with initial conditions
observation, info = env.reset(initial_state='random', options={'pos_bound': 2, 'vel_bound': 0})

# Create a new PPO model
model = PPO(MlpPolicy, env, verbose=1, ent_coef=0.01, tensorboard_log=tensorboard_log_dir)

# Training parameters
num_timesteps = 20_000
num_epochs = 10
start_time = datetime.now()
epoch_count = 0

# Evaluation callback setup
eval_env = Monitor(gym.make("Quadrotor-v0",
                            control_mode='cmd_motor_speeds',
                            quad_params=quad_params,
                            max_time=5,
                            world=world,
                            sim_rate=100,
                            render_mode='None'))

eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                             log_path=log_dir, eval_freq=5000,
                             deterministic=True, render=False)

# Training loop with visualization and monitoring
try:
    all_rewards = []
    while epoch_count < num_epochs:
        # Train the model
        model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False,
                    tb_log_name="PPO-Quad_cmd-motor_" + start_time.strftime('%H-%M-%S'),
                    callback=eval_callback)

        # Save the model
        model_path = os.path.join(models_dir, "PPO", start_time.strftime('%H-%M-%S'), f"hover_{num_timesteps*(epoch_count+1)}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)

        # Evaluate the model
        obs, _ = eval_env.reset()
        done = False
        rewards = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            rewards.append(reward)

        epoch_count += 1
        all_rewards.append(np.sum(rewards))

        # Plotting results for the current epoch
        plt.plot(range(len(rewards)), rewards)
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title(f'Rewards in Epoch {epoch_count}')
        plt.show()
        print(f"Epoch {epoch_count}/{num_epochs} completed, model saved to {model_path}")

except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    model.save(os.path.join(models_dir, "PPO", f"interrupt_{start_time.strftime('%H-%M-%S')}"))

# Plot overall training rewards
plt.figure()
plt.plot(range(1, epoch_count + 1), all_rewards)
plt.xlabel('Epoch')
plt.ylabel('Total Reward')
plt.title('Total Training Reward Over Epochs')
plt.show()
