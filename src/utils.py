import torch
from typing import Iterable
from torch.nn.parameter import Parameter


class ParameterVector():
    def __init__(self, params:Iterable[Parameter]):
        self.params = [p.clone() for p in params]
        self.dim = sum([p.numel() for p in self.params])
    
    def dot(self, v:"ParameterVector") -> float:
        prod = 0
        for p1, p2 in zip(self.params, v.params):
            prod += torch.sum(p1 * p2)
        return prod
    
    def sum(self, v:"ParameterVector", alpha:float=1, inplace:bool=False) -> "ParameterVector":
        v_sum = self if inplace else self.copy()
        for i, p in enumerate(v.params):
            v_sum.params[i] += p*alpha
        return v_sum
    
    def copy(self) -> "ParameterVector":
        params_copy = [p.clone() for p in self.params]
        return ParameterVector(params_copy)

    def norm(self) -> float:
        sum_sq = 0
        for p in self.params:
            sum_sq += torch.sum(p**2)
        return torch.sqrt(sum_sq)
    
    def mult(self, alpha:float, inplace:bool=False) ->"ParameterVector":
        v = self if inplace else self.copy()
        for i, p in enumerate(v.params):
            v.params[i] *= alpha
        return v
    
    def orthonormal(self, vectors:Iterable["ParameterVector"], inplace:bool=False) -> "ParameterVector":
        v_orthonormal = self if inplace else self.copy()
        for v in vectors:
            v_orthonormal.sum(v, alpha=-v_orthonormal.dot(v), inplace=True)
        return v_orthonormal.mult(1/v_orthonormal.norm())
    
    def random_like(params:Iterable[Parameter]):
        params_rand = [Parameter(torch.rand_like(p), requires_grad=p.requires_grad) for p in params]
        return ParameterVector(params_rand)