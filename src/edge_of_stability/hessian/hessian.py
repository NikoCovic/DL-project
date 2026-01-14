from src.edge_of_stability.utils import *
from src.edge_of_stability.preconditioner import Preconditioner
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from typing import Iterable
import numpy as np


class Hessian:
    def __init__(self, model:nn.Module, data, loss_fn, device:str="cpu"):
        self.model = model
        self.device = device
        self.inputs, self.targets = data
        self.loss_fn = loss_fn
        self.dim = sum([p.numel() for p in model.parameters() if p.requires_grad])
        
        self.model.to(self.device) 
        self.update_params()

    def update_params(self):
        self.params = [nn.Parameter(p.detach().clone(), requires_grad=True).to(p.device) for p in self.model.parameters() if p.requires_grad]

    def commutativity_measure(self, preconditioner:Preconditioner, **algo_kwargs):
        # Store the current parameters
        # model_params = self._set_model_params(self.params)

        # Compute loss
        loss = self.loss_fn(self.model(self.inputs.to(self.device)), self.targets.to(self.device))
        # Compute the gradients w.r.t. the loss
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad = torch.autograd.grad(loss, params, create_graph=True)

        def mv(v):
            # Compute PHv - HPv
            Hv = self.hessian_vector_product(v, grad, params, inplace=False)
            PHv = preconditioner.dot(Hv, inplace=True)

            Pv = preconditioner.dot(v, inplace=True)
            HPv = self.hessian_vector_product(Pv, grad, params, inplace=True)

            return params_sum(PHv, HPv, alpha=-1)
        
        def mv_transformed(v):
            # Compute (PH - HP)^Tv = HPv - PHv
            Hv = self.hessian_vector_product(v, grad, params, inplace=False)
            PHv = preconditioner.dot(Hv, inplace=True)

            Pv = preconditioner.dot(v, inplace=True)
            HPv = self.hessian_vector_product(Pv, grad, params, inplace=True)

            return params_sum(HPv, PHv, alpha=-1)
        
        operator = TorchLinearOperator(mv, params)
        operator_transformed = TorchLinearOperator(mv_transformed, params)

        s = spectral_norm(operator, operator_transformed, **algo_kwargs)

        # self._set_model_params(model_params)

        return s

    def eigenvalues(self, preconditioner:Preconditioner=None, **algo_kwargs):
        # Set the model parameters to the current parameters
        # Store the current parameters
        model_params = self._set_model_params(self.params)
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if preconditioner is not None:
            preconditioner_sqrt = preconditioner.pow(0.5)

        # Compute loss
        loss = self.loss_fn(self.model(self.inputs.to(self.device)), self.targets.to(self.device))
        # Fetch the parameters which require a gradient
        params = [p for p in self.model.parameters() if p.requires_grad]
        # Compute the gradient
        grad = torch.autograd.grad(loss, params, create_graph=True)

        # Construct operator for P^{1/2}HP^{1/2}v
        def mv(v):
            Psqv = params_copy(v) if preconditioner is None else preconditioner_sqrt.dot(v)
            HPsqv = self.hessian_vector_product(Psqv, grad, params)
            PsqHPsqv = HPsqv if preconditioner is None else preconditioner_sqrt.dot(HPsqv)
            return PsqHPsqv
        operator = TorchLinearOperator(mv, params)
        stability_constant = preconditioner.frobenius_norm() if preconditioner is not None else None
        eigenvalues, eigenvectors = power_iteration_eigenvalues(operator, stability_constant=stability_constant, **algo_kwargs)
        
        self._set_model_params(model_params)

        return eigenvectors, eigenvalues
    
    def spectral_norm(self, preconditioner:Preconditioner=None, **algo_kwargs):
        # Set the parameters to the current Hessian parameters
        model_params = self._set_model_params(self.params)
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Compute the loss with and gradient
        loss = self.loss_fn(self.model(self.inputs.to(self.device)), self.targets.to(self.device))
        grad = torch.autograd.grad(loss, params, create_graph=True)

        if preconditioner is not None:
            preconditioner_sqrt = preconditioner.pow(0.5)

        # Construct the operators
        # Regular operator is P^{1/2}HP^{1/2}v
        def mv(v:Iterable[Parameter]):
            if preconditioner is not None:
                v = preconditioner.dot(v, inplace=True)#preconditioner_sqrt.dot(v, inplace=True)
            v = self.hessian_vector_product(v, grad, params, inplace=True)
            #if preconditioner is not None:
            #    v = preconditioner_sqrt.dot(v, inplace=True)
            return v
        operator = TorchLinearOperator(mv=mv, params=params)

        # Transposed operator is (P^{1/2}HP^{1/2})^Tv = P^{1/2}HP^{1/2}v
        def mv_transposed(v:Iterable[Parameter]):
            #if preconditioner is not None:
            #    v = preconditioner_sqrt.dot(v, inplace=True)
            v = self.hessian_vector_product(v, grad, params, inplace=True)
            if preconditioner is not None:
                v = preconditioner.dot(v, inplace=True)
            return v
        operator_transposed = TorchLinearOperator(mv=mv_transposed, params=params)

        # Compute the spectral norm
        s = spectral_norm(operator, operator_transposed, **algo_kwargs)
        
        # Reset the model parameters
        self._set_model_params(model_params)

        return s
    
    def update_spectral_norm(self, lr:float, preconditioner:Preconditioner, **algo_kwargs):
        # Set the parameters to the current Hessian parameters
        model_params = self._set_model_params(self.params)
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Compute the loss with and gradient
        loss = self.loss_fn(self.model(self.inputs.to(self.device)), self.targets.to(self.device))
        grad = torch.autograd.grad(loss, params, create_graph=True)

        # Construct the operators
        # Regular operator is (I - \eta PH)v = v - \eta PHv
        def mv(v:Iterable[Parameter]):
            Hv = self.hessian_vector_product(v, grad, params, inplace=False)
            PHv = preconditioner.dot(Hv, inplace=True)
            return params_sum(v, PHv, alpha=-lr)
        operator = TorchLinearOperator(mv=mv, params=params)
        # Trasformed oprator (I - \eta PH)^Tv = (I - \eta HP)v
        def mv_transposed(v:Iterable[Parameter]):
            Pv = preconditioner.dot(v, inplace=False)
            HPv = self.hessian_vector_product(Pv, grad, params)
            return params_sum(v, HPv, alpha=-lr)
        operator_transformed = TorchLinearOperator(mv=mv_transposed, params=params)
        # Compute the spectral norm
        s = spectral_norm(operator, operator_transformed, **algo_kwargs)
        
        # Reset the model parameters
        self._set_model_params(model_params)

        return s
    
    def _set_model_params(self, params:Iterable[Parameter]) -> Iterable[Parameter]:
        current_params = [Parameter(p.data) for p in self.model.parameters() if p.requires_grad]
        for p_m, p in zip([p for p in self.model.parameters() if p .requires_grad], params):
            p_m.data = p.data
        return current_params
    
    def update_eigenvalues(self, lr:float, preconditioner:Preconditioner=None, **algo_kwargs):
        # Since (I - lr*PH) is non-symmetric, it's easier to compute the eigenvalues of (P^{1/2}HP^{1/2})
        # This can be done since \lambda(PH) = \lambda(P^{1/2}HP^{1/2})
        # Additionally, (I - lr*PH)v_i = v_i - lr*PHv_i = (1 - lr*\lambda(PH))v_i, where v_i is an eigenvector
        # So, \lambda_i(I - lr*PH) = 1 - lr*\lambda_i(PH) = 1 - lr*\lambda_i(P^{1/2}HP^{1/2})
        
        _, eigvals = self.eigenvalues(preconditioner=preconditioner, **algo_kwargs)
        eigvals = [1 - lr*e for e in eigvals]

        return eigvals

    def hessian_vector_product(self, v:Iterable[Parameter], grad:Iterable[Parameter], params:Iterable[Parameter], inplace:bool=True) -> Iterable[Parameter]:
        # Reset model gradients
        self.model.zero_grad()

        # Compute the loss and gradient if it is not provided
        if grad is None:
            # Compute loss
            loss = self.loss_fn(self.model(self.inputs.to(self.device)), self.targets.to(self.device))

            # Compute gradient
            grad = torch.autograd.grad(loss, params.params, create_graph=True)
    
        # Compute Hv by differentiating again
        Hv = torch.autograd.grad(params_dot_product(grad, v), params, retain_graph=True)
        v = v if inplace else params_copy(v)
        for p_v, p_Hv in zip(v, Hv):
            p_v.data = p_Hv.data
        return v