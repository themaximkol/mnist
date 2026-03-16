import numpy as np

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


class SGDMomentum:
    def __init__(self, nw_layers, lr, beta=0.9):
        self.layers = nw_layers
        self.lr = lr
        self.beta = beta

        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.vW = np.zeros_like(layer.W)
                layer.vb = np.zeros_like(layer.b)

    def step(self):

        for layer in self.layers:
            if isinstance(layer, Linear):
                # Formula: vW = beta*v_1 + (1-beta)*dW | vb = beta*v_1 + (1-beta)*db

                # Exponential Moving Average, stable version
                # layer.vW = self.beta * layer.vW + (1 - self.beta) * layer.dW
                # layer.vb = self.beta * layer.vb + (1 - self.beta) * layer.db

                # PyTorch version, accumulates the gradient, may explode, lr needs to be reduced
                layer.vW = self.beta * layer.vW + layer.dW
                layer.vb = self.beta * layer.vb + layer.db

                layer.W -= self.lr * layer.vW
                layer.b -= self.lr * layer.vb
