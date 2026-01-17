from .tracker import Tracker
from src.edge_of_stability.hessian import Hessian
from torch.optim import Optimizer
from torch.nn import Module
from src.edge_of_stability.preconditioner_factory import fetch_preconditioner


class UpdateSharpnessTracker(Tracker):
    def __init__(self, hessian:Hessian, optim:Optimizer, model:Module, freq:int=1, n_warmup:int=1):
        super().__init__(freq, n_warmup)
        self.hessian = hessian
        self.optim = optim
        self.model = model

    def _update(self):
        preconditioner = fetch_preconditioner(self.optim, self.model, self.hessian.params)
        lr = self.optim.param_groups[0]["lr"]
        es = self.hessian.update_eigenvalues(lr=lr, preconditioner=preconditioner)
        return abs(es[0])
    
        