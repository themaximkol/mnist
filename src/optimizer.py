from src.layers import Linear

class SGD:
    def __init__(self, nw_layers, lr):
        self.layers = nw_layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db