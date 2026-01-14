from .preconditioner import Preconditioner
from torch.optim import Adam
from typing import Iterable
from torch.nn.parameter import Parameter
import torch
from src.edge_of_stability.utils import *
from torch.optim import Optimizer
from torch.nn import Module


class AdamPreconditioner(Preconditioner):
    def __init__(self, optim:Adam=None, model:Module=None):
        super().__init__(optim, model)

    def compute_p(self, optim:Adam, model:Module):
        params = [p for p in model.parameters() if p.requires_grad]
        self.P_dict = {}
        eps = optim.param_groups[0]["eps"]
        beta1, beta2 = optim.param_groups[0]["betas"]
        for p in params:
            v = optim.state[p]["exp_avg_sq"].detach().clone()
            t = optim.state[p]["step"]
            bias_correction1 = 1 - beta1**t
            bias_correction2 = 1 - beta2**t

            self.P_dict[p] = {}
            self.P_dict[p]["P"] = 1/((torch.sqrt(v/bias_correction2) + eps)*bias_correction1)
            #print("Spectral norm of P ", torch.max(self.P_dict[p]["P"]))

    def copy(self):
        preconditioner_new = AdamPreconditioner()
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
    
    def frobenius_norm(self):
        val = 0
        for p in self.P_dict:
            val += torch.sum(self.P_dict[p]["P"].pow(2)).sqrt()
        return val.cpu().item()
    
    def mul(self, c:float, inplace:bool=False) -> "AdamPreconditioner":
        p = self if inplace else self.copy()
        for p in p.P_dict:
            p.P_dict[p]["P"].mul_(c)
        return p