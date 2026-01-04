from typing import Iterable
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.nn import Module
import torch
from src.utils import params_copy


class Preconditioner:
    def __init__(self, optim:Optimizer, model:Module, p:float):
        self.optim = optim
        self.model = model
        self.p = p
        self.P_dict = None

    def copy(self) -> "Preconditioner":
        pass

    def pow(self, p:float, inplace:bool=False) -> "Preconditioner":
        pass

    def update_params(self):
        self.params = [torch.nn.Parameter(p.clone(), requires_grad=True) for p in self.model.parameters() if p.requires_grad]

    def update(self):
        self.P_dict = {}
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        for p, p_model in zip(self.params, model_params):
            self.P_dict[p] = {}
            self.compute_pow_p(p, p_model)

    def compute_pow_p(self, p:Parameter, p_model:Parameter):
        pass

    def dot(self, v:Iterable[Parameter], inplace:bool=False) -> Iterable[Parameter]:
        pass