from .preconditioner import Preconditioner
from torch.optim import SGD
from typing import Iterable
from torch.nn.parameter import Parameter
from src.utils import ParameterVector


class SGDLRPreconditioner(Preconditioner):
    def __init__(self, sgd_optim:SGD, params:Iterable[Parameter], lr:float, p:float=1):
        self.optim = sgd_optim
        self.p = p
        self.lr = lr
        self.params = params

    def copy(self) -> "SGDLRPreconditioner":
        P_copy = SGDLRPreconditioner(self.optim, self.params, self.lr, p=self.p)
        return P_copy

    def pow(self, p:float, inplace:bool=False) -> "SGDLRPreconditioner":
        P = self if inplace else self.copy()
        P.p = p*self.p
        return P

    def dot(self, v:ParameterVector, inplace:bool=False) -> ParameterVector:
        v = v if inplace else v.copy()
        for p_v in v.params:
            p_v.data = p_v * (self.lr**self.p)
        return v