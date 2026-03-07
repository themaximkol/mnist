import numpy as np


class Network:
    def __init__(self, layers):
        self.layers = layers  # store once at construction

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dX):
        for layer in reversed(self.layers):
            dX = layer.backward(dX)


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


# For training. For inference is used np.argmax(Z, axis=1)
class SoftmaxCEL(Layer):
    def forward(self, Z, y_true):
        self.y_true = y_true

        e = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.y_pred = e / np.sum(e, axis=1, keepdims=True)

        self.cel = -np.sum(y_true * np.log(np.clip(self.y_pred, 1e-9, 1.0))) / len(Z)
        return self.cel

    # the gradient CEL + Softmax together
    def backward(self):
        return (self.y_pred - self.y_true) / len(self.y_true)


class SGD:
    def __init__(self, layers, lr):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db
