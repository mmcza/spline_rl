import torch

from spline_rl.utils.network import AirHockeyNetwork


class ValueNetwork(AirHockeyNetwork):
    def __init__(self, input_space):
        super().__init__(input_space)
        W = 128

        activation = torch.nn.Tanh()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_space.shape[0], W), activation,
            torch.nn.Linear(W, W), activation,
            torch.nn.Linear(W, 1),
        )
    
    def __call__(self, x):
        x, q0, qd, dq0, dqd, ddq0, ddqd = self.prepare_data(x)
        return self.fc(x)