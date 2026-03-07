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

    def save(self, path):
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                weights[f'layer_{i}_W'] = layer.W
                weights[f'layer_{i}_b'] = layer.b
        np.savez(path, **weights)
        print(f"Weights saved to {path}.npz")

    def load(self, path):
        weights = np.load(f"{path}.npz")
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                layer.W = weights[f'layer_{i}_W']
                layer.b = weights[f'layer_{i}_b']
        print(f"Weights loaded from {path}.npz")
