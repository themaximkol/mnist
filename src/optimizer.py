import math

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


class Adam:
    def __init__(self, nw_layers, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Maintains per-parameter learning rates by tracking both gradient
        direction (first moment) and gradient magnitude (second moment).

        :param beta1:     first moment decay rate — how much gradient direction
                          history to retain, default 0.9
        :param beta2:     second moment decay rate — how much gradient magnitude
                          history to retain, default 0.999
        """

        self.layers = nw_layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.mW = np.zeros_like(layer.W)
                layer.vW = np.zeros_like(layer.W)

                layer.mb = np.zeros_like(layer.b)
                layer.vb = np.zeros_like(layer.b)

    def step(self):
        self.t += 1
        for layer in self.layers:
            if isinstance(layer, Linear):
                # first moment — direction
                layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.dW
                layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * layer.db

                # second moment — magnitude
                layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * (layer.dW ** 2)
                layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * (layer.db ** 2)

                # bias correction
                m_hat_W = layer.mW / (1 - self.beta1 ** self.t)
                v_hat_W = layer.vW / (1 - self.beta2 ** self.t)

                m_hat_b = layer.mb / (1 - self.beta1 ** self.t)
                v_hat_b = layer.vb / (1 - self.beta2 ** self.t)

                # update
                layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
                layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
