from enum import Enum
import copy
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from spline_rl.utils.constraints import WallHittingDroneConstraints

from rotorpy.vehicles.crazyflie_params import quad_params

class WallHittingDroneEnv():
    def __init__(self, gamma, horizon, moving_init, interpolation_order, reward_type="end"):
        super().__init__(gamma=gamma, horizon=horizon, interpolation_order=interpolation_order)

        if reward_type = "end":
            self.reward = self.end_reward
        else:
            raise ValueError("Unknown reward type")
        
