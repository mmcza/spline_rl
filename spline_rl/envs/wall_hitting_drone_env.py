from enum import Enum
import copy
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from spline_rl.utils.constraints import WallHittingDroneContraints
import math

from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.world import World
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind
from rotorpy.sensors.imu import Imu
from rotorpy.sensors.external_mocap import MotionCapture

from spline_rl.utils.utils import euler_from_quaternion

class ExitStatus(Enum):
    """ Exit status values indicate the reason for simulation termination. """
    COMPLETE     = 'Success: End reached.'
    TIMEOUT      = 'Timeout: Simulation end time reached.'
    INF_VALUE    = 'Failure: Your controller returned inf motor speeds.'
    NAN_VALUE    = 'Failure: Your controller returned nan motor speeds.'
    OVER_SPEED   = 'Failure: Your quadrotor is out of control; it is going faster than 100 m/s. The Guinness World Speed Record is 73 m/s.'
    OVER_SPIN    = 'Failure: Your quadrotor is out of control; it is spinning faster than 100 rad/s. The onboard IMU can only measure up to 52 rad/s (3000 deg/s).'
    FLY_AWAY     = 'Failure: Your quadrotor is out of control; it flew away with a position error greater than 20 meters.'
    COLLISION    = 'Failure: Your quadrotor collided with an object.'

class WallHittingDroneEnv():
    def __init__(self, gamma = 0,
                       horizon = 0, 
                       interpolation_order = [0], 
                       reward_type = "end",
                       vehicle = Multirotor(quad_params),
                       controller = SE3Control(quad_params),
                       wind_profile = NoWind(),
                       sim_rate = 100,
                       imu = None,
                       mocap = None,
                       estimator = None,
                       world = World.empty((-5, 5, -5, 5, -5, 5)),
                       safety_margin = 0.25):

        self.env_info = dict()

        #self.constraints = WallHittingDroneConstraints(quad_params.rotor_speed_max)

        self.gamma = gamma
        self.horizon = horizon
        self.interpolation_order = interpolation_order

        # Initialize wall in the environment (the wall is as wide and as tall as the environment)
        self.wall = 0
        self.initial_state = dict()

        # Initialize vehicle
        self.vehicle = vehicle

        # Initialize controller
        self.controller = controller

        # Initialize wind profile
        self.wind_profile = wind_profile

        # Initialize simulation frequency
        self.sim_rate = sim_rate

        # Initialize trajectory
        self.trajectory = None
        
        if imu is None:
            # In the event of specified IMU, default to 0 bias with white noise with default parameters as specified below. 
            from rotorpy.sensors.imu import Imu
            self.imu = Imu(p_BS = np.zeros(3,),
                           R_BS = np.eye(3),
                           sampling_rate=sim_rate)
        else:
            self.imu = imu

        if mocap is None:
            # If no mocap is specified, set a default mocap. 
            # Default motion capture properties. Pretty much made up based on qualitative comparison with real data from Vicon. 
            mocap_params = {'pos_noise_density': 0.0005*np.ones((3,)),  # noise density for position 
                    'vel_noise_density': 0.0010*np.ones((3,)),          # noise density for velocity
                    'att_noise_density': 0.0005*np.ones((3,)),          # noise density for attitude 
                    'rate_noise_density': 0.0005*np.ones((3,)),         # noise density for body rates
                    'vel_artifact_max': 5,                              # maximum magnitude of the artifact in velocity (m/s)
                    'vel_artifact_prob': 0.001,                         # probability that an artifact will occur for a given velocity measurement
                    'rate_artifact_max': 1,                             # maximum magnitude of the artifact in body rates (rad/s)
                    'rate_artifact_prob': 0.0002                        # probability that an artifact will occur for a given rate measurement
            }
            from rotorpy.sensors.external_mocap import MotionCapture
            self.mocap = MotionCapture(sampling_rate=sim_rate, mocap_params=mocap_params, with_artifacts=False)
        else:
            self.mocap = mocap

        if estimator is None:
            # In the likely case where an estimator is not supplied, default to the null state estimator. 
            from rotorpy.estimators.nullestimator import NullEstimator
            self.estimator = NullEstimator()
        else:
            self.estimator = estimator

        # Initialize world
        self.world = world
        
        # Set collision detection zone
        self.detection_zone = safety_margin

        # Set safe hitting speed limit
        self.max_hitting_speed = 5.0


        if reward_type == "end":
            self.reward = self.end_reward
        else:
            raise ValueError("Unknown reward type")

    def generate_world(self):
        wall_x = np.random.uniform(-10, -2)
        self.wall = wall_x
        world_map = {"bounds": {"extents": [-10., 10., -10., 10., -0.5, 10.]},
        "blocks": [{"extents": [wall_x, wall_x - 0.5, -10., 10., -0.5, 10.], "color": [1, 0, 0]}]}
        self.world = World(world_map)

        initial_state = dict()
        initial_state['x'] = np.random.uniform(0, 6, (3,))
        initial_state['v'] = np.random.uniform(-2, 3, (3,))
        initial_state['q'] = np.array([0, 0, 0, 1])
        initial_state['w'] = np.random.uniform(-2, 3, (3,))
        initial_state['rotor_speeds'] = np.array([1788.53, 1788.53, 1788.53, 1788.53])
        initial_state['wind']= np.array([0,0,0])
        self.initial_state = initial_state

        return wall_x, initial_state

    def run(self, trajectory, t_max, use_mocap = False):
        if self.wall == 0:
            raise ValueError("World was not generated!\nUse generate_world() before running the simulation")
        
        def sanitize_control_dic(control_dic):
            """
            Return a sanitized version of the control dictionary where all of the elements are np arrays
            """
            control_dic['cmd_motor_speeds'] = np.asarray(control_dic['cmd_motor_speeds'], np.float64).ravel()
            control_dic['cmd_moment'] = np.asarray(control_dic['cmd_moment'], np.float64).ravel()
            control_dic['cmd_q'] = np.asarray(control_dic['cmd_q'], np.float64).ravel()
            return control_dic

        def sanitize_trajectory_dic(trajectory_dic):
            """
            Return a sanitized version of the trajectory dictionary where all of the elements are np arrays
            """
            trajectory_dic['x'] = np.asarray(trajectory_dic['x'], np.float64).ravel()
            trajectory_dic['x_dot'] = np.asarray(trajectory_dic['x_dot'], np.float64).ravel()
            trajectory_dic['x_ddot'] = np.asarray(trajectory_dic['x_ddot'], np.float64).ravel()
            trajectory_dic['x_dddot'] = np.asarray(trajectory_dic['x_dddot'], np.float64).ravel()
            trajectory_dic['x_ddddot'] = np.asarray(trajectory_dic['x_ddddot'], np.float64).ravel()

            return trajectory_dic

        def time_exit(time, t_final):
            """
            Return exit status if the time exceeds t_final, otherwise None.
            """
            if time >= t_final:
                return ExitStatus.TIMEOUT
            return None

        def safety_exit(world, margin, state, flat, control):
            """
            Return exit status if any safety condition is violated, otherwise None.
            """
            if np.any(np.isinf(control['cmd_motor_speeds'])):
                return ExitStatus.INF_VALUE
            if np.any(np.isnan(control['cmd_motor_speeds'])):
                return ExitStatus.NAN_VALUE
            if np.any(np.abs(state['v']) > 100):
                return ExitStatus.OVER_SPEED
            if np.any(np.abs(state['w']) > 100):
                return ExitStatus.OVER_SPIN
            if np.any(np.abs(state['x'] - flat['x']) > 20):
                return ExitStatus.FLY_AWAY

            if len(world.world.get('blocks', [])) > 0:
                # If a world has objects in it we need to check for collisions.  
                collision_pts = world.path_collisions(state['x'], margin)
                no_collision = collision_pts.size == 0
                if not no_collision:
                    return ExitStatus.COLLISION
            return None

        def merge_dicts(dicts_in):
            """
            Concatenates contents of a list of N state dicts into a single dict by
            prepending a new dimension of size N. This is more convenient for plotting
            and analysis. Requires dicts to have consistent keys and have values that
            are numpy arrays.
            """
            dict_out = {}
            for k in dicts_in[0].keys():
                dict_out[k] = []
                for d in dicts_in:
                    dict_out[k].append(d[k])
                dict_out[k] = np.array(dict_out[k])
            return dict_out

        t_step = 1/self.sim_rate

        time    = [0]
        state   = [copy.deepcopy(self.initial_state)]
        state[0]['wind'] = self.wind_profile.update(0, state[0]['x'])
        imu_measurements = []
        mocap_measurements = []
        imu_gt = []
        state_estimate = []
        flat    = [trajectory[0]]
        mocap_measurements.append(self.mocap.measurement(state[-1], with_noise=True, with_artifacts=False))
        if use_mocap:
            # In this case the controller will use the motion capture estimate of the pose and twist for control. 
            control = [sanitize_control_dic(self.controller.update(time[-1], mocap_measurements[-1], flat[-1]))]
        else:
            control = [sanitize_control_dic(self.controller.update(time[-1], state[-1], flat[-1]))]
        state_dot =  self.vehicle.statedot(state[0], control[0], t_step)
        imu_measurements.append(self.imu.measurement(state[-1], state_dot, with_noise=True))
        imu_gt.append(self.imu.measurement(state[-1], state_dot, with_noise=False))
        state_estimate.append(self.estimator.step(state[0], control[0], imu_measurements[0], mocap_measurements[0]))

        exit_status = None

        while True:
            exit_status = exit_status or safety_exit(self.world, self.detection_zone, state[-1], flat[-1], control[-1])
            exit_status = exit_status or time_exit(time[-1], t_max)
            if exit_status:
                break
            time.append(time[-1] + t_step)
            state[-1]['wind'] = self.wind_profile.update(time[-1], state[-1]['x'])
            state.append(self.vehicle.step(state[-1], control[-1], t_step))
            flat.append(sanitize_trajectory_dic(trajectory[math.floor(time[-1]/t_step)]))
            mocap_measurements.append(self.mocap.measurement(state[-1], with_noise=True, with_artifacts=self.mocap.with_artifacts))
            state_estimate.append(self.estimator.step(state[-1], control[-1], imu_measurements[-1], mocap_measurements[-1]))
            if use_mocap:
                control.append(sanitize_control_dic(self.controller.update(time[-1], mocap_measurements[-1], flat[-1])))
            else:
                control.append(sanitize_control_dic(self.controller.update(time[-1], state[-1], flat[-1])))
            state_dot = self.vehicle.statedot(state[-1], control[-1], t_step)
            imu_measurements.append(self.imu.measurement(state[-1], state_dot, with_noise=True))
            imu_gt.append(self.imu.measurement(state[-1], state_dot, with_noise=False))
            if time[-1] + t_step >= t_max:
                break

        time    = np.array(time, dtype=float)    
        state   = merge_dicts(state)
        imu_measurements = merge_dicts(imu_measurements)
        imu_gt = merge_dicts(imu_gt)
        mocap_measurements = merge_dicts(mocap_measurements)
        control         = merge_dicts(control)
        flat            = merge_dicts(flat)
        state_estimate  = merge_dicts(state_estimate)

        reward = self.reward(state)

        return (time, state, control, flat, imu_measurements, imu_gt, mocap_measurements, state_estimate, exit_status, reward)

    def end_reward(self, state):
        r = 0
        
        # Get state from last frame
        x, y, z = state['x'][-1]
        x_dot, y_dot, z_dot = state['v'][-1]
        qx, qy, qz, qw = state['q'][-1]
        roll_dot, pitch_dot, yaw_dot = state['w'][-1]

        # Calculate distance from the wall
        distance_from_wall = abs(x - self.wall) - self.detection_zone
        
        if distance_from_wall < 0.0:
            distance_from_wall = 0.0

        # Calculate speed penalty (prevents from hitting too hard)
        speed_penalty = abs(x_dot - self.max_hitting_speed)

        if speed_penalty < 0.0:
            speed_penalty = 0.0

        # Calculate angles from quaternion
        roll, pitch, yaw = euler_from_quaternion(qx, qy, qz, qw)

        # Calculate reward
        r += np.exp(-10. * distance_from_wall)
        r += np.exp(-2. * abs(z_dot) ** 2) 
        r += np.exp(-2. * abs(pitch) **2)
        r += np.exp(-2. * abs(pitch_dot) ** 2)
        r += np.exp(-2. * speed_penalty ** 2)

        return r

if __name__ == "__main__":
    env = WallHittingDroneEnv()
    wall_x, initial_state = env.generate_world()
    print(initial_state)
    print(wall_x)

    # For testing purposes
    max_t = 5.0
    trajectory = []
    traj_generator = HoverTraj()
    for i in range(5 * 100):
        trajectory.append(traj_generator.update(i/100))

    time, state, control, flat, imu_measurements, imu_gt, mocap_measurements, state_estimate, exit_status, reward = env.run(trajectory, max_t)

    print(state)
    print(reward)


        
