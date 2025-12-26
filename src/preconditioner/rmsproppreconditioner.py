from .preconditioner import Preconditioner
from torch.optim import RMSprop
from typing import Iterable
from torch.nn.parameter import Parameter
import torch
from src.utils import *


class RMSpropPreconditioner(Preconditioner):
    def __init__(self, rmsprop_optim:RMSprop, params:Iterable[Parameter], p:float=1):
        self.optim = rmsprop_optim
        self.p = p
        self.params = params
        self.P_dict = None

    def prepare(self):
        self.P_dict = {}
        for p in self.params:
            # Fetch the square gradients
            M = self.optim.state[p]["square_avg"].clone().detach()

            self.P_dict[p] = {}
            # Store P^p = diag(1/(M + e))^p
            self.P_dict[p]["P_pow_p"] = torch.pow(1/(torch.sqrt(M) + 1e-8), self.p)

    def copy(self) -> "RMSpropPreconditioner":
        P_copy = RMSpropPreconditioner(self.optim, self.params, p=self.p)
        P_copy.P_dict = {}
        for p in self.P_dict:
            P_copy.P_dict[p] = {}
            P_copy.P_dict[p]["P_pow_p"] = self.P_dict[p]["P_pow_p"].clone()
        return P_copy

    def pow(self, p:float, inplace:bool=False) -> "RMSpropPreconditioner":
        P = self if inplace else self.copy()
        P.p = p*self.p
        for param in P.P_dict:
            P.P_dict[param]["P_pow_p"] = torch.pow(self.P_dict[param]["P_pow_p"], P.p)
        return P

    def dot(self, v:Iterable[Parameter], inplace:bool=False) -> Iterable[Parameter]:
        v = v if inplace else params_copy(v)
        for p, p_v in zip(self.P_dict, v):
            p_v.data = self.P_dict[p]["P_pow_p"] * p_v
        return v