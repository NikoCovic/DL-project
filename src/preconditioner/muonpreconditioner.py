import torch
from torch.optim import Muon
from .preconditioner import Preconditioner, PreconditionerNew
from typing import Iterable
from torch.nn.parameter import Parameter
from torch.nn import Module
from src.utils import *


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
                self.P_dict[p]["S"] = S
                # Precompute the power
                # P = (MM^T)^{-1/2} = (USV^TVSU^T)^{-1/2} = (US^2U^T)^{-1/2} = US^{-1}U^T
                # P^p = US^{-p}U^T
                self.P_dict[p]["P"] = U @ torch.diag(S.pow(-1)) @ U.T

    def copy(self):
        preconditioner_new = MuonPreconditioner()
        preconditioner_new.P_dict = {}
        for p in self.P_dict:
            preconditioner_new.P_dict[p] = {}
            preconditioner_new.P_dict[p]["U"] = self.P_dict[p]["U"].detach().clone()
            preconditioner_new.P_dict[p]["S"] = self.P_dict[p]["S"].detach().clone()
        return preconditioner_new
    
    def pow(self, p:float, inplace:bool=False):
        preconditioner_new = self if inplace else self.copy()
        for param in preconditioner_new.P_dict:
            U = preconditioner_new.P_dict[param]["U"]
            S = preconditioner_new.P_dict[param]["S"]
            S_new = S.pow(-p)
            preconditioner_new.P_dict[param]["P"] = U @ torch.diag(S_new) @ U.T
            preconditioner_new.P_dict[param]["S"] = S_new
        return preconditioner_new
    
    def dot(self, v:Iterable[Parameter], inplace:bool=False):
        v = v if inplace else params_copy(v)
        for p_v, p in zip(v, self.P_dict):
            p_v.data = self.P_dict[p]["P"] @ p_v
        return v
            



class MuonPreconditionerNew(Preconditioner):
    def __init__(self, optim:Muon, model:torch.nn.Module, p:float=1, power_method:str="svd"):
        super().__init__(optim, model, p)
        self.power_method = power_method
        self.P_dict = None

    def copy(self) -> "MuonPreconditioner":
        P = MuonPreconditioner(self.optim, self.model, p=self.p, power_method=self.power_method)
        P.P_dict = {}
        for p in self.P_dict:
            P.P_dict[p] = {}
            P.P_dict[p]["U"] = self.P_dict[p]["U"].clone()
            P.P_dict[p]["S"] = self.P_dict[p]["S"].clone()
            P.P_dict[p]["lr"] = self.P_dict[p]["lr"]
            P.P_dict[p]["P_pow_p"] = self.P_dict[p]["P_pow_p"].clone()
        return P
    
    #def update_params(self):
    #    self.params = [torch.nn.Parameter(p.clone(), requires_grad=True) for p in self.model.parameters() if p.requires_grad]

    """
    def update(self):
        #print("Here")
        self.P_dict = {}
        lr = self.optim.param_groups[0]["lr"]
        # Loop through all of the optimizers parameters
        for i, (p, p_model) in enumerate(zip(self.params, self.optim.state.keys())):
            #print("Here 2")
            if self.power_method == "svd":
                # Fetch the corresponding last parmeters
                p_old = self.last_params[i]
                # Reconstruct the true change matrix
                O_hat = (p - p_model)/lr
                # Extract the momentum (and update it with the current gradient)
                g = p.grad.clone().detach()
                M = self.optim.state[p_model]["momentum_buffer"].clone().detach()*0.95 + g
                # Compute the SVD of M = USV^T
                U, Sigma, V = torch.svd(M)
                D = torch.diag(U.T @ O_hat @ V)
                D = torch.clamp_min(D, 1e-12)
                #D = torch.diag(D)
                #print(D)
                # Make sure there are no 0 valued singular values
                Sigma = torch.clamp_min(Sigma, 1e-12)
                S = Sigma * torch.pow(D, -1)
                #print(S)
                # Store the U and S matrices
                self.P_dict[p] = {}
                self.P_dict[p]["U"] = U
                self.P_dict[p]["S"] = S
                # Precompute the power
                # P = (MM^T)^{-1/2} = (USV^TVSU^T)^{-1/2} = (US^2U^T)^{-1/2}  = U|S|^{-1}U^T
                # P^p = U|S|^{-p}U^T
                self.P_dict[p]["P_pow_p"] = U @ torch.diag(S.pow(-self.p)) @ U.T

        #self.last_params = [p.data.clone() for p in self.params]
        self.update_params()
    """

    def compute_pow_p(self, p:Parameter, p_model:Parameter):
        lr = self.optim.param_groups[0]["lr"]
        if self.power_method == "svd":
            # G_t = \nabla_W L(W_t)
            # M_t = \beta M_{t-1} + G_t
            # O_t = orthonormalize(M_t)
            # W_{t+1} = W_t - \eta O_t

            # W_t is p
            # W_{t+1} is p_model

            # Reconstruct the true change matrix O_t = (W_t - W_{t+1})/\eta
            O_hat = (p - p_model)/lr
            # Extract the momentum M_t
            M = self.optim.state[p_model]["momentum_buffer"].clone().detach()
            # Compute the SVD of M = U Sigma V^T
            U, Sigma, Vh = torch.linalg.svd(M)
            D = U.T @ O_hat @ Vh.T
            #print(D)
            D = torch.diag(D)
            D = torch.clamp_min(D, 1e-12)
            #D = torch.diag(D)
            #print(D)
            # Make sure there are no 0 valued singular values
            Sigma = torch.clamp_min(Sigma, 1e-12)
            S = Sigma * torch.pow(D, -1)
            #print(S)
            # Store the U and S matrices
            self.P_dict[p]["U"] = U
            self.P_dict[p]["S"] = S
            self.P_dict[p]["lr"] = lr
            # Precompute the power
            # P = (MM^T)^{-1/2} = (USV^TVSU^T)^{-1/2} = (US^2U^T)^{-1/2} = US^{-1}U^T
            # P^p = US^{-p}U^T
            self.P_dict[p]["P_pow_p"] = U @ torch.diag(((S.pow(-1))).pow(self.p)) @ U.T

    def pow(self, p:float, inplace:bool=False) -> "MuonPreconditioner":
        # Compute the power
        P_pow_p = self if inplace else self.copy()
        P_pow_p.p = self.p * p
        for param in self.P_dict:
            U = P_pow_p.P_dict[param]["U"]
            S = P_pow_p.P_dict[param]["S"]
            lr = P_pow_p.P_dict[param]["lr"]
            P_pow_p.P_dict[param]["P_pow_p"] = U @ torch.diag(((S.pow(-1))).pow(P_pow_p.p)) @ U.T
        return P_pow_p

    def dot(self, v:Iterable[Parameter], inplace:bool=False) -> Iterable[Parameter]:
        v = v if inplace else params_copy(v)
        for p, p_v in zip(self.P_dict.keys(), v):
            #print("before", p_v)
            p_v.data = self.P_dict[p]["P_pow_p"] @ p_v
            #print("dot", p_v)
            #v.params[i] = p_dot
        return v