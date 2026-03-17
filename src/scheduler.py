class ExponentialLR:
    def __init__(self, optimizer, gamma=0.95):
        self.optimizer = optimizer
        self.gamma     = gamma

    def step(self):
        self.optimizer.lr *= self.gamma