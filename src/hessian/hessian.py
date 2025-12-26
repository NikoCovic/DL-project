from src.utils import *
from src.preconditioner import Preconditioner
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from pyhessian import hessian
from typing import Iterable


class Hessian:
    """
    Class for computing the eigencalues of the Hessian/Effective Hessian in Deep Learning problems
    """

    def __init__(self, model:nn.Module, data, loss_fn, device:str="cpu"):
        self.model = model
        self.inputs, self.targets = data
        self.loss_fn = loss_fn
        self.model = model
        self.dim = sum([p.numel() for p in model.parameters() if p.requires_grad])
        self.device = device

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
        
        if preconditioner is not None:
            preconditioner.prepare()
            preconditioner_inv_sqrt = preconditioner.pow(0.5)

        #eigenvectors:list[ParameterVector] = []
        eigenvectors:list[Iterable[Parameter]] = []
        eigenvalues:list[float] = []

        # Compute loss
        loss = self.loss_fn(self.targets.to(self.device), self.model(self.inputs.to(self.device)))
        loss.backward(create_graph=True)
        # Fetch the parameters which require a gradient
        #params = ParameterVector([p for p in self.model.parameters() if p.requires_grad])
        #grad = ParameterVector([0. if p.grad is None else p.grad+0. for p in params.params])
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad = [0. if p.grad is None else p.grad+0. for p in params]

        while len(eigenvectors) < top_n:   

            # Initialize the first eigenvector to a random normalized vector
            #eigenvector:ParameterVector = ParameterVector.random_like([p for p in self.model.parameters() if p.requires_grad])
            #eigenvector.mult(1/eigenvector.norm(), inplace=True)
            eigenvector:Iterable[Parameter] = params_random_like(params)
            params_normalize(eigenvector, inplace=True)

            if method == "power_iteration":
                eigenvalue:float = None

                for i in range(max_iter):
                    # Orthonormalize the eigenvector w.r.t the already computed eigenvectors
                    #eigenvector.orthonormal(eigenvectors, inplace=True)
                    params_orthonormalize(eigenvector, eigenvectors, inplace=True)

                    # Compute P^{-1/2}HP^{-1/2}v or just Hv
                    #v = eigenvector.copy()
                    v = params_copy(eigenvector)
                    if preconditioner is not None:
                        # P^{-1/2}v
                        v = preconditioner_inv_sqrt.dot(v, inplace=True)
                    # Hv
                    v = self.hessian_vector_product(v, grad, params)
                    if preconditioner is not None:
                        # P^{-1/2}v
                        v = preconditioner_inv_sqrt.dot(v, inplace=True)

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
            elif method == "LA":
                # Convert the eigenvector to a flat numpy array
                v0 = []
                for p in eigenvector.params:
                    v0 += p.flatten().tolist()
                v0 = np.array(v0)

                # Create the operator
                def mv(v:np.ndarray):
                    # Reshape to match the eigenvector and create ParameterVector
                    v_torch = []
                    i = 0
                    v = np.astype(v, np.float32)
                    for p in eigenvector.params:
                        _v_p = v[i:i+p.numel()]
                        _v_p = _v_p.reshape(p.shape)
                        _v_p = Parameter(torch.tensor(_v_p).to(self.device), requires_grad=True)
                        v_torch.append(_v_p)
                        i = i+p.numel()
                    v_torch = ParameterVector(v_torch)
                    if preconditioner is not None:
                        # Compute P^{-1/2}v
                        v_torch = preconditioner_inv_sqrt.dot(v_torch)
                    # Compute Hv
                    v_torch = self.hessian_vector_product(v_torch, grad, params)
                    if preconditioner is not None:
                        v_torch = preconditioner_inv_sqrt.dot(v_torch)
                    # Convert back to numpy array
                    _v = []
                    for p in v_torch.params:
                        _v += p.flatten().tolist()
                    _v = np.array(_v)
                    return _v
                
                # Create the linear operator
                n = len(v0)
                op = LinearOperator((n,n), matvec=mv)
                eigenvalues, eigenvectors = eigsh(op, k=top_n, which='LA', v0=v0, tol=tol, maxiter=None)
        
        return eigenvectors, eigenvalues

    def hessian_vector_product(self, v:ParameterVector, grad:ParameterVector, params:ParameterVector, inplace:bool=True) -> ParameterVector:
        # Reset model gradients
        self.model.zero_grad()

        # Compute loss
        #loss = self.loss_fn(self.targets.to(self.device), self.model(self.inputs.to(self.device)))

        # Fetch the parameters which require a gradient
        #params = [p for p in self.model.parameters() if p.requires_grad]

        # Compute gradient
        #grad = torch.autograd.grad(loss, params.params, create_graph=True)
        #grad = ParameterVector(grad)
        
        # Compute the Hessian-vector product by differentiating again
        #Hv = torch.autograd.grad(grad.dot(v), params.params, retain_graph=True)
        #v = v if inplace else v.copy()
        #for p_v, p_Hv in zip(v.params, Hv):
        #    p_v.data = p_Hv.data
        #return v
    
        Hv = torch.autograd.grad(params_dot_product(grad, v), params, retain_graph=True)
        v = v if inplace else params_copy(v)
        for p_v, p_Hv in zip(v, Hv):
            p_v.data = p_Hv.data
        return v