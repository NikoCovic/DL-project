from .frozenoptim import FrozenOptim
import torch
from src.edge_of_stability.utils import params_sum


class FrozenAdamOptim(FrozenOptim):
    def __init__(self, params, preconditioner, lr = 0.001, momentum:float=0.9):
        super().__init__(params, preconditioner, lr, momentum=momentum)

    def step(self):
        params = []
        momentums = []

        for group in self.param_groups:
            momentum = group["momentum"]

            for p in group["params"]:
                g = p.grad

                if g is None:
                    continue

                state = self.state[p]

                params.append(p)

                # Update momentum
                if "exp_avg" not in state.keys():
                    state["exp_avg"] = torch.zeros_like(g)
                m = state["exp_avg"]
                m.mul_(momentum).add_(g, alpha=1-momentum)

                momentums.append(m)

        update = self.preconditioner.dot(momentums, inplace=False)
        params_sum(params, update, alpha=-self.lr, inplace=True)

