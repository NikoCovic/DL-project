from typing import Iterable
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.nn import Module
import torch
from edge_of_stability.utils import params_copy


class Preconditioner:
    def __init__(self, optim:Optimizer=None, model:Module=None):
        if optim is not None and model is not None:
            self.compute_p(optim, model)

    def compute_p(self, optim:Optimizer, model:Module):
        pass

    def copy(self) -> "Preconditioner":
        pass

    def pow(self, p:float, inplace:bool=False) -> "Preconditioner":
        pass

    def dot(self, v:Iterable[Parameter], inplace:bool=False) -> Iterable[Parameter]:
        pass