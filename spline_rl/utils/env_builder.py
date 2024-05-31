from mushroom_rl.core import Logger, MultiprocessEnvironment

from spline_rl.envs.air_hockey_env import AirHockeyEnv
from spline_rl.envs.kinodynamic_cup_env import KinodynamicCupEnv
from spline_rl.envs.wall_hitting_drone_env import WallHittingDroneEnv
from spline_rl.envs.wall_hitting_drone_gym_env import WallHittingDroneEnvGym

def env_builder(env_name, n_envs, env_params):
    env_class = None
    if env_name == "air_hockey":
        env_class = AirHockeyEnv
    elif env_name == "kinodynamic_cup":
        env_class = KinodynamicCupEnv
    elif env_name == "wall_hitting_drone":
        env_class = WallHittingDroneEnv
    elif env_name == "wall_hitting_drone_gym":
        env_class = WallHittingDroneEnvGym
    else:
        raise ValueError("Unknown environment")
    
    env = env_class(**env_params)
    env_info = env.env_info
    if n_envs > 1:
        env = MultiprocessEnvironment(env_class, n_envs=n_envs, **env_params)
    return env, env_info
