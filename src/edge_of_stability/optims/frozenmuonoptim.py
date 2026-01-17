from .frozenoptim import FrozenOptim
from src.edge_of_stability.utils import params_sum
import torch


class FrozenMuonOptim(FrozenOptim):
    def __init__(self, params, preconditioner, lr=0.001, momentum:float=0.9, weight_decay:float=0, nesterov:bool=False):
        super().__init__(params, preconditioner, lr, 
                         momentum=momentum, 
                         weight_decay=weight_decay, 
                         nesterov=nesterov)

    def step(self):
        params = []
        momentums = []
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                # Store the parameters into the list
                params.append(p)

                state = self.state[p]
                # Update momentum
                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                
                # Add weight decay
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

                # Store the momentum
                momentums.append(buf)

        update = self.preconditioner.dot(momentums, inplace=False)
        
        params_sum(params, update, alpha=-self.lr, inplace=True)

        #print(params)
