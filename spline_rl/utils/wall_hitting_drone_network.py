import torch

from spline_rl.utils.utils import unpack_data_drone


class WallHittingDroneNetwork(torch.nn.Module):
    def __init__(self, input_space):
        super(WallHittingDroneNetwork, self).__init__()
        self.input_space = input_space

    def normalize_input(self, x):
        low = torch.Tensor(self.input_space.low)[None]
        high = torch.Tensor(self.input_space.high)[None]
        normalized = (x - low) / (high - low)
        normalized = 2 * normalized - 1
        return normalized

    def prepare_data(self, x):
        q0, qd, dq0, dqd, ddq0, ddqd = unpack_data_drone(x)
        x = self.normalize_input(x)
        return x, q0, qd, dq0, dqd, ddq0, ddqd

    
        

class WallHittingDroneConfigurationTimeNetwork(WallHittingDroneNetwork):
    def __init__(self, input_shape, output_shape, input_space):
        super(WallHittingDroneConfigurationTimeNetwork, self).__init__(input_space)

        activation = torch.nn.Tanh()
        W = 256
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
        )

        self.q_est = torch.nn.Sequential(
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, output_shape[0])#, activation,
        )

        self.t_est = torch.nn.Sequential(
            torch.nn.Linear(W, output_shape[1]),
        )

    def __call__(self, x):
        x, q0, qd, dq0, dqd, ddq0, ddqd = self.prepare_data(x)

        x = self.fc(x)
        q_prototype = self.q_est(x)
        ds_dt_prototype = self.t_est(x)
        return torch.cat([q_prototype, ds_dt_prototype], dim=-1)

class WallHittingDroneConfigurationTimeNetworkWrapper(WallHittingDroneConfigurationTimeNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(WallHittingDroneConfigurationTimeNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"])


class WallHittingDroneLogSigmaNetwork(WallHittingDroneNetwork):
    def __init__(self, input_shape, output_shape, input_space, init_sigma):
        super(WallHittingDroneLogSigmaNetwork, self).__init__(input_space)

        self._init_sigma = init_sigma

        activation = torch.nn.Tanh()
        W = 128
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, output_shape[0]),
        )

    def __call__(self, x):
        x, q0, qd, dq0, dqd, ddq0, ddqd = self.prepare_data(x)
        #x = torch.log(self._init_sigma)[None].to(torch.float64)
        x = self.fc(x) + torch.log(self._init_sigma)[None]
        return x

class WallHittingDroneLogSigmaNetworkWrapper(WallHittingDroneLogSigmaNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(WallHittingDroneLogSigmaNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"], params["init_sigma"])

class WallHittingDroneConfigurationNetwork(WallHittingDroneNetwork):
    def __init__(self, input_shape, output_shape, input_space):
        super(WallHittingDroneConfigurationNetwork, self).__init__(input_space)

        activation = torch.nn.Tanh()
        W = 256
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, output_shape[0])#, activation,
        )

    def __call__(self, x):
        x, q0, qd, dq0, dqd, ddq0, ddqd = self.prepare_data(x)
        x = self.fc(x)
        return x

class WallHittingDroneConfigurationNetworkWrapper(WallHittingDroneConfigurationNetwork):
    def __init__(self, input_shape, output_shape, params, **kwargs):
        super(WallHittingDroneConfigurationNetworkWrapper, self).__init__(input_shape, output_shape, params["input_space"])