import numpy as np

from src.layers import BatchNorm


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

    def save(self, path):
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                weights[f'layer_{i}_W'] = layer.W
                weights[f'layer_{i}_b'] = layer.b
            if isinstance(layer, BatchNorm):
                weights[f'layer_{i}_gamma'] = layer.gamma
                weights[f'layer_{i}_beta'] = layer.beta
                weights[f'layer_{i}_running_mean'] = layer.running_mean
                weights[f'layer_{i}_running_var'] = layer.running_var
        np.savez(path, **weights)
        print(f"Weights saved to {path}.npz")

    def load(self, path):
        weights = np.load(f"{path}.npz")
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                layer.W = weights[f'layer_{i}_W']
                layer.b = weights[f'layer_{i}_b']
            if isinstance(layer, BatchNorm):
                layer.gamma = weights[f'layer_{i}_gamma']
                layer.beta = weights[f'layer_{i}_beta']
                layer.running_mean = weights[f'layer_{i}_running_mean']
                layer.running_var = weights[f'layer_{i}_running_var']
        print(f"Weights loaded from {path}.npz")
