class Tracker:
    def __init__(self, freq:int, n_warmup:int):
        self.time = 0
        self.measurements = []
        self.freq = freq
        self.n_warmup = n_warmup

    def _update(self):
        pass

    def update(self):
        if self.time >= self.n_warmup and self.time % self.freq == 0:
            self._update()
        self.time += 1