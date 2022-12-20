import torch
from torch import nn

# Neural Network in Renyi paper
class RenyiNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        out = self.linear(x)
        return out


# Neural Network in FedFair paper
class FedFairNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        out = self.linear(x)
        return out
    