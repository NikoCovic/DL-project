import torch.nn as nn
import torch

class SquareLinear(nn.Module):
    def __init__(self, in_features, out_features, bias:bool=True):
        super().__init__()
        in_features = in_features+1 if bias else in_features
        self.layer = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        x = torch.cat((x, torch.ones((len(x), 1))), 1)
        return self.layer(x)