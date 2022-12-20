import numpy as np
import torch
from torch import nn

class RenyiLogisticRegression(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(RenyiLogisticRegression, self).__init__()
        self.layer = nn.Linear(input_size, output_size, bias=False)  # No bias them in LogisticRegression case
        self.sigmoid = nn.Sigmoid()
        torch.nn.init.zeros_(self.layer.weight)  # The weight parameter is initialized to zero

    def forward(self, X):
        out = self.layer(X)
        out = self.sigmoid(out)
        return out


if __name__ == '__main__':
    # Testing the Pytorch code.

    x = torch.tensor(np.random.rand(64, 29028), dtype=torch.float16)
    net = RenyiLogisticRegression(29028)
    y = net(x)
    print(y)