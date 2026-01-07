import torch
from typing import Iterable
from torch.nn.parameter import Parameter


class ParameterVector():
    def __init__(self, params:Iterable[Parameter]):
        self.params = [p for p in params]
        self.dim = sum([p.numel() for p in self.params])
    
    def dot(self, v:"ParameterVector") -> float:
        prod = 0
        for p1, p2 in zip(self.params, v.params):
            prod += torch.sum(p1 * p2)
        return prod
    
    def sum(self, v:"ParameterVector", alpha:float=1, inplace:bool=False) -> "ParameterVector":
        v_sum = self if inplace else self.copy()
        for i, p in enumerate(v.params):
            v_sum.params[i].data.add_(p*alpha)
        return v_sum
    
    def copy(self) -> "ParameterVector":
        params_copy = [p.clone() for p in self.params]
        return ParameterVector(params_copy)

    def norm(self) -> float:
        return self.dot(self)**0.5
    
    def mult(self, alpha:float, inplace:bool=False) ->"ParameterVector":
        v = self if inplace else self.copy()
        for i, p in enumerate(v.params):
            v.params[i].mul_(alpha)
        return v
    
    def orthonormal(self, vectors:Iterable["ParameterVector"], inplace:bool=False) -> "ParameterVector":
        v_orthonormal = self if inplace else self.copy()
        for v in vectors:
            v_orthonormal.sum(v, alpha=-v_orthonormal.dot(v), inplace=True)
        return v_orthonormal.mult(1/v_orthonormal.norm(), inplace=True)
    
    def random_like(params:Iterable[Parameter]):
        params_rand = [Parameter(torch.rand_like(p), requires_grad=False) for p in params]
        return ParameterVector(params_rand)
    

def params_copy(params:Iterable[Parameter]):
    return [Parameter(p.clone(), requires_grad=p.requires_grad) for p in params]


def params_dot_product(params1:Iterable[Parameter], params2:Iterable[Parameter]):
    return sum([torch.sum(p1 * p2) for p1, p2 in zip(params1, params2)])


def params_norm(params:Iterable[Parameter]):
    return torch.sqrt(params_dot_product(params, params))


def params_scale(params:Iterable[Parameter], scalar:float, inplace:bool=False):
    params = params if inplace else params_copy(params)
    for p in params:
        p.data.mul_(scalar)
    return params


def params_sum(params1:Iterable[Parameter], params2:Iterable[Parameter], alpha:float=1, inplace:bool=False):
    params_sum = params1 if inplace else params_copy(params1)
    for p1, p2 in zip(params_sum, params2):
        p1.data.add_(p2*alpha)
    return params_sum


def params_normalize(params:Iterable[Parameter], inplace:bool=False):
    params = params if inplace else params_copy(params)
    norm = params_norm(params)
    return params_scale(params, 1/(norm+1e-8), inplace=True)


def params_orthonormalize(params:Iterable[Parameter], other_params:Iterable[Iterable[Parameter]], inplace:bool=False):
    params = params if inplace else params_copy(params)
    for v in other_params:
        params = params_sum(params, v, alpha=-params_dot_product(params, v), inplace=True)
    return params_normalize(params, inplace=True)


def params_random_like(params:Iterable[Parameter]):
    return [Parameter(torch.rand_like(p), requires_grad=False) for p in params]


class TorchLinearOperator():
    def __init__(self, mv, params):
        self.mv = mv
        self.params = params

    def dot(self, v:Iterable[Parameter], inplace=False) -> Iterable[Parameter]:
        v = v if inplace else params_copy(v)
        return self.mv(v)


def power_iteration_eigenvalues(operator:TorchLinearOperator, top_n:int=1, max_iter:int=200, tol:float=1e-8):
    eigenvectors = []
    eigenvalues = []

    while len(eigenvectors) < top_n:
        # Initalize eigenvector
        v = params_random_like(operator.params)
        # Initialize eigenvalue to None
        eigval = None

        for _ in range(max_iter):
            # Orthonormalize v w.r.t the currently computed eigenvectors
            v = params_orthonormalize(v, eigenvectors, inplace=True)
            
            # Compute Mv
            Mv = operator.dot(v, inplace=False)

            # Compute the eigenvalue
            temp_eigval = params_dot_product(v, Mv).cpu().item()

            # Compute the eigenvector by normalizing Mv
            v = params_normalize(Mv, inplace=True)

            # Check if the tolerance threshold has been reached, if so, break the loop
            if eigval is None:
                eigval = temp_eigval
            else:
                if abs(eigval - temp_eigval) / (abs(eigval) + 1e-6) < tol:
                    break
                else:
                    eigval = temp_eigval

        eigenvalues.append(eigval)
        eigenvectors.append(v)

    return eigenvalues, eigenvectors


def spectral_norm(operator:TorchLinearOperator, operator_transformed:TorchLinearOperator, max_iter:int=200, tol:float=1e-8):
    v = params_random_like(operator.params)
    v = params_normalize(v, inplace=True)

    s = None
    
    for _ in range(max_iter):
        # Compute u_k = Av_{k-1}
        u = operator.dot(v, inplace=False)
        # Compute u_k = u_k / ||u_k||_2
        u = params_normalize(u, inplace=True)

        # Compute v_k = A^Tu_k
        v = operator_transformed.dot(u, inplace=True)
        # Compute v_k = v_k / ||v_k||_2
        v = params_normalize(v, inplace=True)

        # Compute s = u_k^T A v_k and check if it beats the tolerance threshold
        s_temp = params_dot_product(u, operator.dot(v)).cpu().item()#params_norm(operator.dot(v)).cpu().item()
        if s is None:
            s = s_temp
        else:
            if abs(s - s_temp) / (abs(s) + 1e-6) < tol:
                break
            else:
                s = s_temp

    return s
        

