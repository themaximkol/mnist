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


class BatchNorm(Layer):
    def __init__(self, n_in, epsilon=1e-5, rho=0.9):
        self.gamma = np.ones(n_in)
        self.beta = np.zeros(n_in)

        self.epsilon = epsilon
        self.rho = rho
        self.training = True

        # running stats — used at inference
        self.running_mean = np.zeros(n_in)
        self.running_var = np.ones(n_in)

    def forward(self, X):
        if not self.training:
            X_hat = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * X_hat + self.beta

        self.N = X.shape[0]

        self.mean = np.sum(X, axis=0) / self.N
        self.variance = np.sum(np.square(X - self.mean), axis=0) / self.N

        self.X_hat = (X - self.mean) / np.sqrt(self.variance + self.epsilon)

        self.running_mean = self.rho * self.running_mean + (1 - self.rho) * self.mean
        self.running_var = self.rho * self.running_var + (1 - self.rho) * self.variance

        self.X = X

        return self.gamma * self.X_hat + self.beta

    def backward(self, dY):
        # gradients for learnable params
        self.dgamma = np.sum(dY * self.X_hat, axis=0)
        self.dbeta = np.sum(dY, axis=0)

        # gradient w.r.t normalized input
        dX_hat = dY * self.gamma

        # gradient w.r.t variance — path through normalization
        dvar = np.sum(
            dX_hat * (self.X - self.mean) * -0.5 * (self.variance + self.epsilon) ** -1.5,
            axis=0
        )

        # gradient w.r.t mean — path through normalization and variance
        dmean = np.sum(dX_hat * -1 / np.sqrt(self.variance + self.epsilon), axis=0)

        # combine all three paths into dX
        dX = (dX_hat / np.sqrt(self.variance + self.epsilon) +
              dvar * 2 * (self.X - self.mean) / self.N +
              dmean / self.N)

        return dX
