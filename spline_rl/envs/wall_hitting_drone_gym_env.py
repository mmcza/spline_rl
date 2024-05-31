import warnings

import gymnasium as gym
from gymnasium import spaces as gym_spaces
import numpy as np

try:
    import pybullet_envs
    pybullet_found = True
except ImportError:
    pybullet_found = False

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils.spaces import *
from mushroom_rl.utils.viewer import ImageViewer

from spline_rl.utils.utils import euler_from_quaternion

from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.world import World
import math
from spline_rl.utils.constraints import WallHittingDroneContraints

gym.logger.set_level(40)

class WallHittingDroneEnvGym(Environment):
    def __init__(self, horizon=None, gamma=0.99, headless = False, wrappers=None, wrappers_args=None,
                 **env_args):
        """
        Constructor.

        Args:
             name (str): gym id of the environment;
             horizon (int): the horizon. If None, use the one from Gym;
             gamma (float, 0.99): the discount factor;
             headless (bool, False): If True, the rendering is forced to be headless.
             wrappers (list, None): list of wrappers to apply over the environment. It
                is possible to pass arguments to the wrappers by providing
                a tuple with two elements: the gym wrapper class and a
                dictionary containing the parameters needed by the wrapper
                constructor;
            wrappers_args (list, None): list of list of arguments for each wrapper;
            ** env_args: other gym environment parameters.

        """

        # MDP creation
        self._not_pybullet = True
        self._first = True
        self._headless = headless
        self._viewer = None
        if pybullet_found and '- ' + name in pybullet_envs.getList():
            import pybullet
            pybullet.connect(pybullet.DIRECT)
            self._not_pybullet = False

        world_map = {"bounds": {"extents": [-10., 10., -10., 10., -0.5, 10.]},
        "blocks": [{"extents": [-5, -5.5, -10., 10., -0.5, 10.], "color": [1, 0, 0]}]}
        
        world = World(world_map)

        self.env = gym.make("Quadrotor-v0", 
                        control_mode ='cmd_motor_speeds', 
                        reward_fn = self.end_reward,
                        quad_params = quad_params,
                        max_time = 5,
                        world = world,
                        sim_rate = 100,
                        #render_mode='rgb_array',
                        render_mode = None,
                        render_fps=30)
        
        if wrappers is not None:
            if wrappers_args is None:
                wrappers_args = [dict()] * len(wrappers)
            for wrapper, args in zip(wrappers, wrappers_args):
                if isinstance(wrapper, tuple):
                    self.env = wrapper[0](self.env, *args, **wrapper[1])
                else:
                    self.env = wrapper(self.env, *args, **env_args)

        horizon = self._set_horizon(self.env, horizon)

        # MDP properties
        assert not isinstance(self.env.observation_space,
                              gym_spaces.MultiDiscrete)
        assert not isinstance(self.env.action_space, gym_spaces.MultiDiscrete)

        dt = self.env.unwrapped.dt if hasattr(self.env.unwrapped, "dt") else 0.1
        action_space = self._convert_gym_space(self.env.action_space)
        observation_space = self._convert_gym_space(self.env.observation_space)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)


        # Drone Controller
        self.controller = SE3Control(quad_params)

        # Flag to only give the reward once - at the moment of hitting wall
        self.reward_given = False

        self.constraints = WallHittingDroneContraints(q_dot_max= np.array([100]*6))
        self.state = None

        interpolation_order = 5

        self.env_info = dict()
        self.env_info['rl_info'] = mdp_info
        self.env_info['dt'] = dt
        #self.env_info['rl_info'] = dict()
        #self.env_info['rl_info']['observation_space'] = observation_space
        #self.env_info['rl_info']['constraints'] = dict()
        #self.env_info['rl_info']['constraints'] = self.constraints
        #self.env_info['rl_info']['interpolation_order'] = interpolation_order
        #self.env_info['rl_info']['constraints']['constraints_num'] = 9

        super().__init__(mdp_info)

    # Create reward function - weights can be adjusted
    def end_reward(self, observation, action, weights = {'pitch': 1, 'pitch_dot': 1, 'z_dot': 1, 'x_dot': 1}):
        r = 0
        if not self.reward_given:
            if abs(observation[0] + 5.0) < 0.05:
                self.reward_given = True
                roll, pitch, yaw = euler_from_quaternion(observation[6:10])
                pitch_dot = observation[11]
                x_dot = observation[3]
                z_dot = observation[5]
                
                # Calculate rewards
                r += weights['pitch']*(1-math.tanh(abs(pitch)))
                r += weights['pitch_dot']*(1-math.tanh(abs(pitch_dot)))
                r += weights['x_dot']*(1-math.tanh(abs(x_dot - 5.0)))
                r += weights['z_dot']*(1-math.tanh(abs(z_dot)))

        return r

    def reset(self, state=None):
        self.reward_given = False
        if state is None:
            state, info = self.env.reset()
            self.state = state
            return np.atleast_1d(state), info
        else:
            _, info = self.env.reset()
            self.env.state = state
            self.state = state

            return np.atleast_1d(state), info

    def _convert_action(self, action):
        obs = self.state
        state = {'x': obs[0:3], 'v': obs[3:6],
                 'q': obs[6:10], 'w': obs[10:14]}
        print(state)
        print(action)

        # The line below is if action is like this: [x, x_dot, x_ddot]
        # flat = {'x': action[0, 0:3], 'x_dot': action[1, 0:3], 'x_ddot': action[2, 0:3],
        #         'x_dddot': [0, 0, 0], 'yaw': action[0, 5], 'yaw_dot': action[1, 5], 'yaw_ddot': action[2, 5]}

        flat = {'x': action[0, 0:3], 'x_dot': action[1, 0:3], 'x_ddot': np.array([0, 0, 0]),
                'x_dddot': [0, 0, 0], 'yaw': action[0, 5], 'yaw_dot': action[1, 5], 'yaw_ddot': np.array([0])}        

        # If action is x, x_dot, x_ddot, t, dt
        # x, x_dot, x_ddot, t, dt = action
        # flat = {'x': x[0:3], 'x_dot': x_dot[0:3], 'x_ddot': x_ddot[0:3],
        #         'x_dddot': [0, 0, 0], 'yaw': x[5], 'yaw_dot': x_dot[5], 'yaw_ddot': x_ddot[5]}

        # If there is no good way to get time than we can set t to 0
        t = 0
        control_dict = self.controller.update(t, state, flat)
        cmd_motor_speeds = control_dict['cmd_motor_speeds']
        action_ = np.interp(cmd_motor_speeds, [self.env.unwrapped.rotor_speed_min, self.env.unwrapped.rotor_speed_max], [-1,1])
        return action_


    def step(self, action):
        action = self._convert_action(action)
        obs, reward, absorbing, _, info = self.env.step(action) #truncated flag is ignored 
        self.state = obs

        return np.atleast_1d(obs), reward, absorbing, info

    def render(self, record=False):
        if self._first or self._not_pybullet:
            img = self.env.render()

            if self._first:
                self._viewer =  ImageViewer((img.shape[1], img.shape[0]), self.info.dt, headless=self._headless)

            self._viewer.display(img)

            self._first = False

            if record:
                return img
            else:
                return None

        return None

    def stop(self):
        try:
            if self._not_pybullet:
                self.env.close()
                
                if self._viewer is not None:
                    self._viewer.close()
        except:
            pass

    @staticmethod
    def _set_horizon(env, horizon):

        while not hasattr(env, '_max_episode_steps') and env.env != env.unwrapped:
                env = env.env

        if horizon is None:
            if not hasattr(env, '_max_episode_steps'):
                raise RuntimeError('This gymnasium environment has no specified time limit!')
            horizon = env._max_episode_steps
            if horizon == np.inf:
                warnings.warn("Horizon can not be infinity.")
                horizon = int(1e4)

        if hasattr(env, '_max_episode_steps'):
            env._max_episode_steps = horizon

        return horizon

    @staticmethod
    def _convert_gym_space(space):
        if isinstance(space, gym_spaces.Discrete):
            return Discrete(space.n)
        elif isinstance(space, gym_spaces.Box):
            return Box(low=space.low, high=space.high, shape=space.shape)
        else:
            raise ValueError