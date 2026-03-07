import numpy as np
from src.layers import Layer


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
