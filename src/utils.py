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
    return [p.clone() for p in params]


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
