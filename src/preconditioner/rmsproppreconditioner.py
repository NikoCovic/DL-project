from .preconditioner import Preconditioner
from torch.optim import RMSprop
from typing import Iterable
from torch.nn.parameter import Parameter
import torch
from src.utils import *


class RMSpropPreconditioner(Preconditioner):
    def __init__(self, optim:RMSprop, model:torch.nn.Module, p:float=1):
        super().__init__(optim, model, p)
    
    """
    def prepare(self):
        self.P_dict = {}
        for p in self.params:
            # Fetch the square gradients
            g = p.grad.clone().detach()
            M = self.optim.state[p]["square_avg"].clone().detach()*0.99 + 0.01*g*g

            self.P_dict[p] = {}
            # Store P^p = diag(1/(M + e))^p
            self.P_dict[p]["P_pow_p"] = torch.pow(1/(torch.sqrt(M) + 1e-8), self.p)
    """

    def compute_pow_p(self, p, p_model):
        lr = self.optim.param_groups[0]["lr"]
        # Fetch the square gradients
        #g = p.grad.clone().detach()
        M = self.optim.state[p_model]["square_avg"].clone().detach()#*0.99 + 0.01*g*g

        #self.P_dict[p] = {}
        # Store P^p = diag(1/(M + e))^p
        self.P_dict[p]["P_pow_p"] = torch.pow(lr/(torch.sqrt(M) + 1e-8), self.p)

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
            P.P_dict[param]["P_pow_p"] = torch.pow(self.P_dict[param]["P_pow_p"], p)
        return P

    def dot(self, v:Iterable[Parameter], inplace:bool=False) -> Iterable[Parameter]:
        v = v if inplace else params_copy(v)
        for p, p_v in zip(self.P_dict, v):
            p_v.data.mul_(self.P_dict[p]["P_pow_p"])#self.P_dict[p]["P_pow_p"])
        return v