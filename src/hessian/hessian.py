from src.utils import *
from src.preconditioner import Preconditioner
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from typing import Iterable
from pyhessian import hessian


class Hessian:
    """
    Class for computing the eigencalues of the Hessian/Effective Hessian in Deep Learning problems
    """

    def __init__(self, model:nn.Module, data, loss_fn, device:str="cpu"):
        self.model = model
        self.inputs, self.targets = data
        self.update_params()
        self.loss_fn = loss_fn
        self.model = model
        self.dim = sum([p.numel() for p in model.parameters() if p.requires_grad])
        self.device = device

    def update_params(self):
        self.params = [nn.Parameter(p.detach().clone(), requires_grad=True).to(p.device) for p in self.model.parameters() if p.requires_grad]

    def commutativity_measure(self, preconditioner:Preconditioner, **algo_kwargs):
        # Store the current parameters
        model_params = [nn.Parameter(p.clone()) for p in self.model.parameters() if p.requires_grad]
        for p_m, p in zip([p for p in self.model.parameters() if p.requires_grad], self.params):
            if p.requires_grad:
                p_m.data = p.data

        # Compute loss
        loss = self.loss_fn(self.targets.to(self.device), self.model(self.inputs.to(self.device)))
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

        # Reset the model parameters
        for p_m, p_old in zip([p for p in self.model.parameters() if p.requires_grad], model_params):
            if p_m.requires_grad:
                p_m.data = p_old.data

        return s

    def eigenvalues(self, max_iter:int=200, tol:float=1e-12, top_n:int=1, preconditioner:Preconditioner=None, method:str="power_iteration"):
        """
        Computes the `top_n` eigenvalues of the Hessian or Preconditioned Hessian

        :param max_iter: The maximum number of power iteration steps to do
        :type max_iter: int
        :param tol: The tolerance for stopping the eigenvalue computation algorithms
        :type tol: float
        :param top_n: The number of top eigenvalues to compute
        :type top_n: int
        :param preconditioner: The preconditioner to use, specify None is no preconditioner should be used, default None
        :type preconditioner: Preconditioner
        :param method: The method to use to compute the eigenvalues, one of 'power_iteration' or 'LA' (experimental)
        :type method: str
        """

        # Set the model parameters to the current parameters
        # Store the current parameters
        model_params = [nn.Parameter(p.clone()) for p in self.model.parameters() if p.requires_grad]
        for p_m, p in zip([p for p in self.model.parameters() if p.requires_grad], self.params):
            if p.requires_grad:
                p_m.data = p.data

        # Update the preconditioner parameters to the current ones
        #preconditioner.params = self.params
        
        if preconditioner is not None:
            #preconditioner.prepare()
            preconditioner_sqrt = preconditioner.pow(0.5)

        #eigenvectors:list[ParameterVector] = []
        #eigenvectors:list[Iterable[Parameter]] = []
        #eigenvalues:list[float] = []

        # Compute loss
        loss = self.loss_fn(self.targets.to(self.device), self.model(self.inputs.to(self.device)))
        # Compute the gradients w.r.t. the loss
        #loss.backward(create_graph=True)
        # Fetch the parameters which require a gradient
        #params = ParameterVector([p for p in self.model.parameters() if p.requires_grad])
        #grad = ParameterVector([0. if p.grad is None else p.grad+0. for p in params.params])
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad = torch.autograd.grad(loss, params, create_graph=True)
        #grad = [0. if p.grad is None else p.grad+0. for p in params]

        if method == "power_iteration":
            """
            # Compute the eigenvectors
            while len(eigenvectors) < top_n:   

                # Initialize the first eigenvector to a random normalized vector
                #eigenvector:ParameterVector = ParameterVector.random_like([p for p in self.model.parameters() if p.requires_grad])
                #eigenvector.mult(1/eigenvector.norm(), inplace=True)
                eigenvector:Iterable[Parameter] = params_random_like(params)
                params_normalize(eigenvector, inplace=True)

                eigenvalue:float = None

                for i in range(max_iter):
                    # Orthonormalize the eigenvector w.r.t the already computed eigenvectors
                    #eigenvector.orthonormal(eigenvectors, inplace=True)
                    params_orthonormalize(eigenvector, eigenvectors, inplace=True)

                    # Compute P^{-1/2}HP^{-1/2}v or just Hv
                    #v = eigenvector.copy()
                    v = params_copy(eigenvector)
                    if preconditioner is not None:
                        # P^{1/2}v
                        v = preconditioner_sqrt.dot(v, inplace=True)
                    # Hv
                    v = self.hessian_vector_product(v, grad, params)
                    if preconditioner is not None:
                        # P^{1/2}v
                        v = preconditioner_sqrt.dot(v, inplace=True)
                        #v = preconditioner.dot(v, inplace=True)

                    # Compute the eigenvalue
                    #temp_eigenvalue = v.dot(eigenvector).cpu().item()
                    temp_eigenvalue = params_dot_product(v, eigenvector).cpu().item()

                    # Compute the eigenvector by normalizing
                    #norm = v.norm().cpu().item()
                    #eigenvector = v.mult(1/norm, inplace=True)
                    eigenvector = params_normalize(v, inplace=True)

                    if eigenvalue is None:
                        eigenvalue = temp_eigenvalue
                    else:
                        if abs(eigenvalue - temp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                            break
                        else:
                            eigenvalue = temp_eigenvalue

                eigenvalues.append(eigenvalue)
                eigenvectors.append(eigenvector)
                """
            # Construct operator for P^{1/2}HP^{1/2}v
            def mv(v):
                Psqv = params_copy(v) if preconditioner is None else preconditioner_sqrt.dot(v)
                HPsqv = self.hessian_vector_product(Psqv, grad, params)
                PsqHPsqv = HPsqv if preconditioner is None else preconditioner_sqrt.dot(HPsqv)
                return PsqHPsqv
            operator = TorchLinearOperator(mv, params)
            eigenvalues, eigenvectors = power_iteration_eigenvalues(operator, max_iter=max_iter, top_n=top_n, tol=tol)
        elif method == "LA":
            # Initialize a random eigenvector and normalize
            eigenvector:Iterable[Parameter] = params_random_like(params)
            params_normalize(eigenvector, inplace=True)

            # Convert the eigenvector to a flat numpy array
            v0 = []
            for p in eigenvector:
                v0 += p.flatten().tolist()
            v0 = np.array(v0)

            # Create the operator
            def mv(v:np.ndarray):
                # Reshape to match the eigenvector and create Parameters
                v_torch = []
                i = 0
                v = np.astype(v, np.float32)
                for p in eigenvector:
                    _v_p = v[i:i+p.numel()]
                    _v_p = _v_p.reshape(p.shape)
                    _v_p = Parameter(torch.tensor(_v_p).to(self.device), requires_grad=True)
                    v_torch.append(_v_p)
                    i = i+p.numel()
                #v_torch = ParameterVector(v_torch)
                if preconditioner is not None:
                    # Compute P^{-1/2}v
                    v_torch = preconditioner_sqrt.dot(v_torch, inplace=True)
                # Compute Hv
                v_torch = self.hessian_vector_product(v_torch, grad, params)
                if preconditioner is not None:
                    v_torch = preconditioner_sqrt.dot(v_torch, inplace=True)
                # Convert back to numpy array
                _v = []
                for p in v_torch:
                    _v += p.flatten().tolist()
                _v = np.array(_v)
                return _v
            
            # Create the linear operator
            n = len(v0)
            op = LinearOperator((n,n), matvec=mv)
            eigenvalues, eigenvectors = eigsh(op, k=top_n, which='LA', v0=v0, tol=tol, maxiter=None)
        
        # Reset the model parameters
        for p_m, p_old in zip([p for p in self.model.parameters() if p.requires_grad], model_params):
            if p_m.requires_grad:
                p_m.data = p_old.data

        return eigenvectors, eigenvalues
    
    def spectral_norm(self, max_iter:int=200, tol=1e-8):
        # Set the parameters to the current Hessian parameters
        model_params = [nn.Parameter(p.clone()) for p in self.model.parameters() if p.requires_grad]
        for p_m, p in zip([p for p in self.model.parameters() if p.requires_grad], self.params):
            if p.requires_grad:
                p_m.data = p.data
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Compute the loss with and gradient
        loss = self.loss_fn(self.targets.to(self.device), self.model(self.inputs.to(self.device)))
        grad = torch.autograd.grad(loss, params, create_graph=True)

        # Construct the operators
        # Regular operator is Hv and so is the transformed one since H^T = H
        def mv(v:Iterable[Parameter]):
            Hv = self.hessian_vector_product(v, grad, inplace=True)
            return Hv
        operator = TorchLinearOperator(mv=mv, params=params)

        # Compute the spectral norm
        s = spectral_norm(operator, operator, max_iter=max_iter, tol=tol)
        
        # Reset the model parameters
        for p_m, p_old in zip([p for p in self.model.parameters() if p.requires_grad], model_params):
            if p_m.requires_grad:
                p_m.data = p_old.data

        return s
    
    def update_spectral_norm(self, lr:float, preconditioner:Preconditioner=None, max_iter:int=200, tol:float=1e-8):
        # Set the parameters to the current Hessian parameters
        model_params = [nn.Parameter(p.clone()) for p in self.model.parameters() if p.requires_grad]
        for p_m, p in zip([p for p in self.model.parameters() if p.requires_grad], self.params):
            if p.requires_grad:
                p_m.data = p.data
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Compute the loss with and gradient
        loss = self.loss_fn(self.targets.to(self.device), self.model(self.inputs.to(self.device)))
        grad = torch.autograd.grad(loss, params, create_graph=True)

        # Construct the operators
        # Regular operator is (I - \eta PH)v = v - \eta PHv
        def mv(v:Iterable[Parameter]):
            Hv = self.hessian_vector_product(v, grad, params, inplace=False)
            PHv = Hv if preconditioner is None else preconditioner.dot(Hv, inplace=True)
            return params_sum(v, PHv, alpha=-lr)
        operator = TorchLinearOperator(mv=mv, params=params)
        # Trasformed oprator (I - \eta PH)^Tv = (I - \eta HP)v
        def mv_transformed(v:Iterable[Parameter]):
            Pv = params_copy(v) if preconditioner is None else preconditioner.dot(v, inplace=False)
            HPv = self.hessian_vector_product(Pv, grad, params)
            return params_sum(v, HPv, alpha=-lr)
        operator_transformed = TorchLinearOperator(mv=mv_transformed, params=params)
        # Compute the spectral norm
        s = spectral_norm(operator, operator_transformed, max_iter=max_iter, tol=tol)
        
        # Reset the model parameters
        for p_m, p_old in zip([p for p in self.model.parameters() if p.requires_grad], model_params):
            if p_m.requires_grad:
                p_m.data = p_old.data

        return s
    
    def update_eigenvalues(self, lr:float, top_n:int=1, preconditioner:Preconditioner=None, max_iter:int=200, tol:float=1e-8):
        # Set the parameters to the current Hessian parameters
        model_params = [nn.Parameter(p.clone()) for p in self.model.parameters() if p.requires_grad]
        for p_m, p in zip([p for p in self.model.parameters() if p.requires_grad], self.params):
            if p.requires_grad:
                p_m.data = p.data
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Compute the loss with and gradient
        loss = self.loss_fn(self.targets.to(self.device), self.model(self.inputs.to(self.device)))
        grad = torch.autograd.grad(loss, params, create_graph=True)

        # Since (I - PH) is non-symmetric, it's easier to compute the eigenvalues of (P^{1/2}HP^{1/2})
        # This can be done since \lambda(PH) = \lambda(P^{1/2}HP^{1/2})
        # Additionally, (I - PH)v_i = v_i - PHv_i = (1 - \lambda(PH))v_i, where v_i is an eigenvector
        # So, \lambda_i(I - PH) = 1 - \lambda_i(PH) = 1 - \lambda_i(P^{1/2}HP^{1/2})
        
        _, eigvals = self.eigenvalues(max_iter=max_iter, tol=tol, preconditioner=preconditioner, top_n=top_n, method="power_iteration")
        eigvals = [1 - lr*e for e in eigvals]
        
        # Reset the model parameters
        for p_m, p_old in zip([p for p in self.model.parameters() if p.requires_grad], model_params):
            if p_m.requires_grad:
                p_m.data = p_old.data

        return eigvals

    def hessian_vector_product(self, v:Iterable[Parameter], grad:Iterable[Parameter], params:Iterable[Parameter], inplace:bool=True) -> Iterable[Parameter]:
        # Reset model gradients
        self.model.zero_grad()

        # Compute the loss and gradient if it is not provided
        if grad is None:
            # Compute loss
            loss = self.loss_fn(self.targets.to(self.device), self.model(self.inputs.to(self.device)))

            # Compute gradient
            grad = torch.autograd.grad(loss, params.params, create_graph=True)
    
        # Compute Hv by differentiating again
        Hv = torch.autograd.grad(params_dot_product(grad, v), params, retain_graph=True)
        v = v if inplace else params_copy(v)
        for p_v, p_Hv in zip(v, Hv):
            p_v.data = p_Hv.data
        return v