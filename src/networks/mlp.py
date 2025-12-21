import torch
import torch.nn as nn
import numpy as np
from .utils import get_activation


class MLP(nn.Module):
    def __init__(self, input_shape, output_dim,
                 activation:str="gelu",
                 width:int=256,
                 bias:bool=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), width, bias=bias),
            get_activation(activation),
            nn.Linear(width, width, bias=bias),
            get_activation(activation),
            nn.Linear(width, output_dim, bias=bias)
        )

    def forward(self, x):
        return self.net(x)