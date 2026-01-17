from .preconditioner import Preconditioner
from torch.optim import RMSprop
from typing import Iterable
from torch.nn.parameter import Parameter
import torch
from src.edge_of_stability.utils import *
from torch.nn import Module


class RMSpropPreconditioner(Preconditioner):
    def __init__(self, optim:RMSprop=None, model:Module=None):
        super().__init__(optim, model, None)

    def compute_p(self, optim:RMSprop, model:Module, params_old:Iterable[Parameter]=None):
        params = [p for p in model.parameters() if p.requires_grad]
        self.P_dict = {}
        eps = optim.param_groups[0]["eps"]
        for p in params:
            S = optim.state[p]["square_avg"].detach().clone()
            self.P_dict[p] = {}
            self.P_dict[p]["P"] = 1/(torch.sqrt(S) + eps)
            #print("Spectral norm of P ", torch.max(self.P_dict[p]["P"]))

    def copy(self):
        preconditioner_new = RMSpropPreconditioner()
        preconditioner_new.P_dict = {}
        for p in self.P_dict:
            preconditioner_new.P_dict[p] = {}
            preconditioner_new.P_dict[p]["P"] = self.P_dict[p]["P"].detach().clone()
        return preconditioner_new

    def pow(self, p:float, inplace:bool=False):
        preconditioner_p = self if inplace else self.copy()
        for param in preconditioner_p.P_dict:
            preconditioner_p.P_dict[param]["P"].pow_(p)
        return preconditioner_p

    def dot(self, v:Iterable[Parameter], inplace:bool=False):
        v = v if inplace else params_copy(v)
        for p_v, p in zip(v, self.P_dict):
            p_v.data.mul_(self.P_dict[p]["P"])
        return v
    
    def mul(self, c:float, inplace:bool=False) -> "RMSpropPreconditioner":
        p = self if inplace else self.copy()
        for p in p.P_dict:
            p.P_dict[p]["P"].mul_(c)
        return p
    
    def frobenius_norm(self):
        val = 0
        for p in self.P_dict:
            val += torch.sum(self.P_dict[p]["P"].pow(2)).sqrt()
        return val.cpu().item()