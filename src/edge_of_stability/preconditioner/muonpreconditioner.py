import torch
from torch.optim import Muon
from .preconditioner import Preconditioner
from typing import Iterable
from torch.nn.parameter import Parameter
from torch.nn import Module
from src.edge_of_stability.utils import *
import math


class MuonPreconditioner(Preconditioner):
    def __init__(self, optim:Muon=None, model:Module=None, params_old:Iterable[Parameter]=None):
        super().__init__(optim, model, params_old)

    def compute_p(self, optim:Muon, model:Module, params_old:Iterable[Parameter]=None):
        self.P_dict = {}
        lr = optim.param_groups[0]["lr"]
        params = [p for p in model.parameters() if p.requires_grad]
        for i, p in enumerate(params):
            self.P_dict[p] = {}
            # Extract the momentum M_t
            M = optim.state[p]["momentum_buffer"].detach().clone()
            #M_norm = M/torch.norm(M)
            # Compute the SVD of M = U Sigma V^T
            U, Sigma, Vh = torch.linalg.svd(M.type(torch.float32))
            if params_old is None:
                S = Sigma
            else:
                O = (params_old[i] - p) / lr
                D = torch.diag(U.T @ O @ Vh.T)
                D = D.clamp_min(1e-8)
                S = Sigma * D.pow(-1)
                #print("D", D, "S", S, "Sigma", Sigma)
            # Compute a singular value threshold
            #s_thresh = 1e-3#0.05 * torch.median(S)
            #S = S[S > s_thresh]
            #U = U[:, :len(S)]
            #S = S.type(M.dtype)
            #U = U.type(M.dtype)
            #print("S", S, "s_thresh", s_thresh)
            # Make sure there are no 0 valued singular values
            #S = S + 1e-6#torch.clamp_min(S, 1e-3)
            S = S.pow(-1)#/torch.max(S.pow(-1))
            #print(S)
            #print(torch.min(S))
            # Store the U and S matrices
            self.P_dict[p]["U"] = U.type(M.dtype)
            self.P_dict[p]["S"] = S.type(M.dtype)
            # Precompute the power
            # P = (MM^T)^{-1/2} = (USV^TVSU^T)^{-1/2} = (US^2U^T)^{-1/2} = US^{-1}U^T
            # P^p = US^{-p}U^T
            self.P_dict[p]["P"] = (U @ torch.diag(S) @ U.T).type(M.dtype)
            #print("Spectral norm of P ", torch.max(S))

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
            S.pow_(p)
            preconditioner_new.P_dict[param]["P"] = U @ torch.diag(S) @ U.T
        return preconditioner_new
    
    def dot(self, v:Iterable[Parameter], inplace:bool=False):
        v = v if inplace else params_copy(v)
        for p_v, p in zip(v, self.P_dict):
            P = self.P_dict[p]["P"]
            p_v.data = P @ p_v
        return v
    
    def frobenius_norm(self):
        val = 0
        for p in self.P_dict:
            val += torch.sum(self.P_dict[p]["P"].pow(2)).sqrt()
        return val.cpu().item()
    
    def mul(self, c:float, inplace:bool=False) -> "MuonPreconditioner":
        p = self if inplace else self.copy()
        for p in p.P_dict:
            p.P_dict[p]["P"].mul_(c)
        return p