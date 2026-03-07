import numpy as np


class Layer:
    def forward(self, X):
        raise NotImplementedError

    def backward(self, dX):
        raise NotImplementedError


class ReLU(Layer):
    def forward(self, X):
        self.X = X
        return np.maximum(0, self.X)

    def backward(self, dX):
        return (self.X > 0) * dX


class Linear(Layer):
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)
        self.b = np.zeros(n_out)

    def forward(self, X):
        self.X = X
        return self.X @ self.W + self.b

    def backward(self, dX):
        self.dW = self.X.T @ dX
        self.db = np.sum(dX, axis=0)

        return dX @ self.W.T  # dx (delta) passed to the previous layers
