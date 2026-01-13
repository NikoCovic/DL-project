import torch
from torch.optim import Muon
from .preconditioner import Preconditioner
from typing import Iterable
from torch.nn.parameter import Parameter
from torch.nn import Module
from edge_of_stability.utils import *


class MuonPreconditioner(Preconditioner):
    def __init__(self, optim:Muon=None, model:Module=None):
        super().__init__(optim, model)

    def compute_p(self, optim:Muon, model:Module):
        if optim is not None and model is not None:
            self.P_dict = {}
            params = [p for p in model.parameters() if p.requires_grad]
            for p in params:
                self.P_dict[p] = {}
                # Extract the momentum M_t
                M = optim.state[p]["momentum_buffer"].detach().clone()
                # Compute the SVD of M = U Sigma V^T
                U, S, Vh = torch.linalg.svd(M)
                # Make sure there are no 0 valued singular values
                S = torch.clamp_min(S, 1e-12)
                # Store the U and S matrices
                self.P_dict[p]["U"] = U
                self.P_dict[p]["S"] = S.pow(-1)
                # Precompute the power
                # P = (MM^T)^{-1/2} = (USV^TVSU^T)^{-1/2} = (US^2U^T)^{-1/2} = US^{-1}U^T
                # P^p = US^{-p}U^T
                self.P_dict[p]["P"] = U @ torch.diag(S) @ U.T

    def copy(self):
        preconditioner_new = MuonPreconditioner()
        preconditioner_new.P_dict = {}
        for p in self.P_dict:
            preconditioner_new.P_dict[p] = {}
            preconditioner_new.P_dict[p]["U"] = self.P_dict[p]["U"].detach().clone()
            preconditioner_new.P_dict[p]["S"] = self.P_dict[p]["S"].detach().clone()
            preconditioner_new.P_dict[p]["P"] = self.P_dict[p]["P"].detach().clone()
        return preconditioner_new
    
    def pow(self, p:float, inplace:bool=False):
        preconditioner_new = self if inplace else self.copy()
        for param in preconditioner_new.P_dict:
            U = preconditioner_new.P_dict[param]["U"]
            S = preconditioner_new.P_dict[param]["S"]
            S_new = S.pow(p)
            preconditioner_new.P_dict[param]["P"] = U @ torch.diag(S_new) @ U.T
            preconditioner_new.P_dict[param]["S"] = S_new
        return preconditioner_new
    
    def dot(self, v:Iterable[Parameter], inplace:bool=False):
        v = v if inplace else params_copy(v)
        for p_v, p in zip(v, self.P_dict):
            p_v.data = self.P_dict[p]["P"] @ p_v
        return v