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

        # Velocities for SGD with momentum
        self.vW = None
        self.vb = None

    def forward(self, X):
        self.X = X
        return self.X @ self.W + self.b

    def backward(self, dX):
        self.dW = self.X.T @ dX
        self.db = np.sum(dX, axis=0)

        return dX @ self.W.T  # dx (delta) passed to the previous layers


class Dropout(Layer):
    def __init__(self, p, training=True, ):
        self.p = p
        self.training = training

    def forward(self, X):
        if not self.training:
            return X

        self.mask = np.random.choice([0, 1], size=X.shape, p=[self.p, 1 - self.p])
        return X * self.mask * (1 / (1 - self.p))

    def backward(self, dX):
        if not self.training:
            return dX

        return dX * self.mask * (1 / (1 - self.p))
