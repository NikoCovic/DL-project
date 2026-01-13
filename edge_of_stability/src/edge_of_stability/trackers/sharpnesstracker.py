from .tracker import Tracker
from edge_of_stability.hessian import Hessian


class SharpnessTracker(Tracker):
    def __init__(self, hessian:Hessian, freq:int=1, n_warmup:int=5):
        super().__init__(freq, n_warmup)
        self.hessian = hessian

    def _update(self):
        _, es = self.hessian.eigenvalues(top_n=1)
        return es[0]
