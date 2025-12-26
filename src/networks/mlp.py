import torch
import torch.nn as nn
import numpy as np
from .utils import get_activation


class MLP(nn.Module):
    def __init__(self, input_shape, output_dim,
                 activation:str="gelu",
                 n_hidden:int=2,
                 width:int=256,
                 bias:bool=True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(np.prod(input_shape), width, bias=bias)
        self.activation = get_activation(activation)
        self.hidden = [nn.Linear(width, width, bias=bias) for _ in range(n_hidden)]
        self.output_layer = nn.Linear(width, output_dim, bias=bias)

    def forward(self, x):
        x = self.activation(self.input_layer(self.flatten(x)))
        for h in self.hidden:
            x = self.activation(h(x))
        return self.output_layer(x)