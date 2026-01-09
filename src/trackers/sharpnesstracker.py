from .tracker import Tracker
from src.hessian import Hessian


class SharpnessTracker(Tracker):
    def __init__(self, hessian:Hessian, freq:int=1, n_warmup:int=5):
        super().__init__(freq, n_warmup)
        self.hessian = hessian

    def _update(self):
        _, es = self.hessian.eigenvalues(top_n=1)
        e = es[0]
        self.measurements.append(e)
