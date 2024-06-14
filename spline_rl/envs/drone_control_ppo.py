import gymnasium as gym
import numpy as np
import os
from datetime import datetime
from scipy.spatial.transform import Rotation as R

from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.world import World

try:
    import stable_baselines3
except:
    raise ImportError('To run this example you must have Stable Baselines installed via pip install stable_baselines3')

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
# Define directories for saving models and logs
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "logs")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

num_cpu = 4 # Number of CPUs for parallel processing (if applicable)

# Custom environment subclassing the QuadrotorEnv
class CustomQuadrotorEnv(QuadrotorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewarded = False # Track if reward has been given
  # Custom reward function
    def compute_reward(self, state, action):
        if np.abs(state[0] - 5.0) < 0.05 and not self.rewarded:
            self.rewarded = True
            quaternion = state[6:10]
            euler_angles = R.from_quat(quaternion).as_euler('xyz', degrees=False)
            pitch = euler_angles[0]
            print("Success!")
            return 500 - np.exp(state[4]-5.0) * 0.001 - np.exp(abs(pitch)) * 0.05 - np.exp(abs(state[10])) * 0.01
        else:
            if self.rewarded:
                return 0
            else:
                if state[0] < -5:
                    x = -5
                else:
                    x = state[0]
                return -np.exp(5.0 - x) * 0.001 
# Define the world map with boundaries
world_map = {"bounds": {"extents": [-10., 10., -10., 10., -10, 10.]}}
world = World(world_map)
# custom environment
env = CustomQuadrotorEnv(
    control_mode ='cmd_motor_speeds', 
    quad_params = quad_params,
    max_time = 5,
    world = world,
    sim_rate = 100,
    render_mode='None'
)

env.reward_fn = env.compute_reward  # Set the reward function to the instance method
# Reset the environment with a random initial state
observation, info = env.reset(initial_state='random', options={'pos_bound': 2, 'vel_bound': 0})
# Create the PPO model
model = PPO(MlpPolicy, env, verbose=1, ent_coef=0.01, tensorboard_log=log_dir)

num_timesteps = 20_000 # Number of timesteps per learning iteration
num_epochs = 10 # Total number of learning epochs

start_time = datetime.now() # Record the start time

epoch_count = 0  # Initialize epoch counter
while True:
       # Train the model for a specified number of timesteps
    model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False, tb_log_name="PPO-Quad_cmd-motor_"+start_time.strftime('%H-%M-%S'))
    model.save(f"{models_dir}/PPO/{start_time.strftime('%H-%M-%S')}/hover_{num_timesteps*(epoch_count+1)}")
    epoch_count += 1  # Increment the epoch counter
