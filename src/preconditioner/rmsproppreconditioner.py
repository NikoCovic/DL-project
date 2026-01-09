from .preconditioner import Preconditioner, PreconditionerNew
from torch.optim import RMSprop
from typing import Iterable
from torch.nn.parameter import Parameter
import torch
from src.utils import *
from torch.optim import Optimizer
from torch.nn import Module


class RMSpropPreconditioner(Preconditioner):
    def __init__(self, optim:Optimizer=None, model:Module=None):
        super().__init__(optim, model)

    def compute_p(self, optim:Optimizer, model:Module):
        params = [p for p in model.parameters() if p.requires_grad]
        self.P_dict = {}
        for p in params:
            S = optim.state[p]["square_avg"].detach().clone()
            self.P_dict[p] = {}
            self.P_dict[p]["P"] = 1/(torch.sqrt(S) + 1e-8)

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


class RMSpropPreconditionerNew(Preconditioner):
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
            P.P_dict[param]["P_pow_p"] = torch.pow(self.P_dict[param]["P_pow_p"], p)
        return P

    def dot(self, v:Iterable[Parameter], inplace:bool=False) -> Iterable[Parameter]:
        v = v if inplace else params_copy(v)
        for p, p_v in zip(self.P_dict, v):
            p_v.data.mul_(self.P_dict[p]["P_pow_p"])#self.P_dict[p]["P_pow_p"])
        return v