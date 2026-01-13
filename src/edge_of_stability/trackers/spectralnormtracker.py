from .tracker import Tracker
from src.edge_of_stability.hessian import Hessian


class SpectralNormTracker(Tracker):
    def __init__(self, hessian:Hessian, freq:int=1, n_warmup:int=5):
        super().__init__(freq, n_warmup)
        self.hessian = hessian

    def _update(self):
        s = self.hessian.spectral_norm()
        return s