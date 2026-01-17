from torch.optim import Optimizer


class FrozenOptim(Optimizer):
    def __init__(self, params, preconditioner, lr:float=1e-3, **kwargs):

        self.preconditioner = preconditioner

        self.params = list(params)

        self.lr = lr

        defaults = kwargs
        defaults["lr"] = lr
        super().__init__(self.params, defaults)

    def step(self):
        pass