class EigOptimizer:
    def __init__(self, optims):
        self.optims = optims
        if not isinstance(self.optims, list):
            self.optims = [optims]
    
    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()
        
    def step(self):
        for optim in self.optims:
            optim.step()
