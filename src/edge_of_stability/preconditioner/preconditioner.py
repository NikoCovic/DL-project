from typing import Iterable
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.nn import Module


class Preconditioner:
    def __init__(self, optim:Optimizer=None, model:Module=None, params_old:Iterable[Parameter]=None):
        if optim is not None and model is not None:
            self.compute_p(optim, model, params_old)

    def compute_p(self, optim:Optimizer, model:Module, params_old:Iterable[Parameter]=None):
        pass

    def copy(self) -> "Preconditioner":
        pass

    def pow(self, p:float, inplace:bool=False) -> "Preconditioner":
        pass

    def dot(self, v:Iterable[Parameter], inplace:bool=False) -> Iterable[Parameter]:
        pass

    def mul(self, c:float, inplace:bool=False) -> "Preconditioner":
        pass

    def frobenius_norm(self) -> float:
        pass