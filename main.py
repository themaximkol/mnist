import numpy as np


class Network:
    def __init__(self, layers):
        self.layers = layers  # store once at construction

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X


class Layer:
    def forward(self, X):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class ReLU(Layer):
    def forward(self, X):
        self.X = X
        return np.maximum(0, self.X)


class Linear(Layer):
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out)
        self.b = np.zeros(n_out)

    def forward(self, X):
        return X @ self.W + self.b


class Softmax(Layer):
    def forward(self, X):
        e = np.exp(X - np.max(X, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
