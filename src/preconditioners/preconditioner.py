from torch.optim.optimizer import Optimizer
from torch.optim import Muon
import torch
import numpy as np
from torch import Tensor
from typing import Iterable


class PreconditionedOptimizer():
    def __init__(self, optimizers:Iterable[torch.optim.Optimizer]):
        self.optimizers = list(optimizers)

    def step(self):
        for optim in self.optimizers:
            optim.step()

    def zero_grad(self):
        for optim in self.optimizers:
            optim.zero_grad()

    def pow_vector_product(self, pow, v):
        pass


def _muon_p_inverse_v(optim:Muon, v:np.array):
    for group in optim.param_groups:
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]

        params_with_grad: list[Tensor] = []
        grads: list[Tensor] = []
        muon_momentum_bufs: list[Tensor] = []

        has_complex = optim._init_group(
            group,
            params_with_grad,
            grads,
            muon_momentum_bufs,
        )

        # Need to compute P^(-1) = (G^TG)^(1/2) via Newton-Schulz