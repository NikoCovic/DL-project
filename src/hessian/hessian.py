from src.utils import ParameterVector
from src.preconditioner import Preconditioner
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch
from scipy.sparse.linalg import LinearOperator, eigsh


class Hessian:
    def __init__(self, model:nn.Module, data, loss_fn, device:str="cpu"):
        self.model = model
        self.inputs, self.targets = data
        self.loss_fn = loss_fn
        self.model = model
        self.dim = sum([p.numel() for p in model.parameters() if p.requires_grad])
        self.device = device

    def eigenvalues(self, max_iter:int=200, tol:float=1e-12, top_n:int=1, preconditioner:Preconditioner=None, method:str="power_iteration"):
        if preconditioner is not None:
            preconditioner.prepare()
            preconditioner_inv_sqrt = preconditioner.pow(0.5)

        eigenvectors:list[ParameterVector] = []
        eigenvalues:list[float] = []

        while len(eigenvectors) < top_n:   

            # Initialize the first eigenvector to a random normalized vector
            eigenvector:ParameterVector = ParameterVector.random_like([p for p in self.model.parameters() if p.requires_grad])
            eigenvector.mult(1/eigenvector.norm(), inplace=True)

            if method == "power_iteration":
                eigenvalue:float = None

                for i in range(max_iter):
                    # Orthonormalize the eigenvector w.r.t the already computed eigenvectors
                    eigenvector.orthonormal(eigenvectors, inplace=True)

                    # Compute P^{-1/2}HP^{-1/2}v or just Hv
                    v = eigenvector.copy()
                    if preconditioner is not None:
                        # P^{-1/2}v
                        v = preconditioner_inv_sqrt.dot(v, inplace=True)
                    # Hv
                    v = self.hessian_vector_product(v)
                    if preconditioner is not None:
                        # P^{-1/2}v
                        v = preconditioner_inv_sqrt.dot(v, inplace=True)

                    # Compute the eigenvalue
                    temp_eigenvalue = v.dot(eigenvector).cpu().item()

                    # Compute the eigenvector by normalizing
                    norm = v.norm().cpu().item()
                    eigenvector = v.mult(1/norm, inplace=True)

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
                        v_torch = preconditioner.pow(-0.5).dot(v_torch)
                    # Compute Hv
                    v_torch = self.hessian_vector_product(v_torch)
                    if preconditioner is not None:
                        v_torch = preconditioner.pow(-0.5).dot(v_torch)
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

    def hessian_vector_product(self, v:ParameterVector, inplace:bool=True) -> ParameterVector:
        # Reset model gradients
        self.model.zero_grad()

        # Compute loss
        loss = self.loss_fn(self.targets.to(self.device), self.model(self.inputs.to(self.device)))

        # Fetch the parameters which require a gradient
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Compute gradient
        grad = torch.autograd.grad(loss, params, create_graph=True)
        grad = ParameterVector(grad)
        
        # Compute the Hessian-vector product by differentiating again
        Hv = torch.autograd.grad(grad.dot(v), params, retain_graph=True)
        v = v if inplace else v.copy()
        for p_v, p_Hv in zip(v.params, Hv):
            p_v.data = p_Hv.data
        return v