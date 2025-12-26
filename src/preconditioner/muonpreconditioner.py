import torch
from torch.optim import Muon
from .preconditioner import Preconditioner
from typing import Iterable
from torch.nn.parameter import Parameter
from src.utils import *


class MuonPreconditioner(Preconditioner):
    def __init__(self, muon_optim:Muon, params:Iterable[Parameter], p:float=1, power_method:str="svd"):
        #super().__init__()
        self.optim = muon_optim
        self.params = list(params)
        self.power_method = power_method
        self.p = p
        self.P_dict = None

    def copy(self) -> "MuonPreconditioner":
        P = MuonPreconditioner(self.optim, self.params, p=self.p, power_method=self.power_method)
        P.P_dict = {}
        for p in self.P_dict:
            P.P_dict[p] = {}
            P.P_dict[p]["U"] = self.P_dict[p]["U"].clone()
            P.P_dict[p]["S"] = self.P_dict[p]["S"].clone()
            P.P_dict[p]["P_pow_p"] = self.P_dict[p]["P_pow_p"].clone()
        return P
    
    def prepare(self):
        self.P_dict = {}
        # Loop through all of the optimizers parameters
        for p in self.params:
            # Extract the momentum
            M = self.optim.state[p]['momentum_buffer'].clone().detach()
            
            if self.power_method == "svd":
                # Compute the SVD of M = USV^T
                U, S, V = torch.svd(M)
                # Make sure eigen-values are at least 1e-12
                S = torch.clamp_min(S, 1e-12)
                # Store the U and S matrices
                self.P_dict[p] = {}
                self.P_dict[p]["U"] = U
                self.P_dict[p]["S"] = S
                # Precompute the power
                # P = (MM^T)^{1/2} = (USV^TVSU^T)^{1/2} = (US^2U^T)^{1/2}  = USU^T
                # P^p = US^pU^T
                self.P_dict[p]["P_pow_p"] = U @ torch.diag(S.pow(-self.p)) @ U.T

    def pow(self, p:float, inplace:bool=False) -> "MuonPreconditioner":
        # Compute the power
        P_pow_p = self if inplace else self.copy()
        P_pow_p.p = self.p * p
        for param in self.P_dict:
            U = P_pow_p.P_dict[param]["U"]
            S = P_pow_p.P_dict[param]["S"]
            P_pow_p.P_dict[param]["P_pow_p"] = U @ torch.diag(S.pow(self.p)) @ U.T
        return P_pow_p

    def dot(self, v:Iterable[Parameter], inplace:bool=False) -> Iterable[Parameter]:
        v = v if inplace else params_copy(v)
        for p, p_v in zip(self.P_dict.keys(), v):
            #print("before", p_v)
            p_v.data = self.P_dict[p]["P_pow_p"] @ p_v
            #print("dot", p_v)
            #v.params[i] = p_dot
        return v