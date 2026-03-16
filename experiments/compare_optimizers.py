import matplotlib.pyplot as plt
import numpy as np

from src.layers import Linear, ReLU
from src.network import Network
from src.loss import SoftmaxCEL
from src.optimizer import SGD, SGDMomentum, Adam
from src.data_loader import X_train, y_train


def train(optimizer_class, optimizer_kwargs, epochs=10, batch_size=32):
    # same architecture, same seed so weights start identical
    np.random.seed(42)
    layers = [
        Linear(784, 128), ReLU(),
        Linear(128, 128), ReLU(),
        Linear(128, 64), ReLU(),
        Linear(64, 10)]

    # layers = [
    #     Linear(784, 32), ReLU(),
    #     Linear(32, 32), ReLU(),
    #     Linear(32, 32), ReLU(),
    #     Linear(32, 10)
    # ]
    network = Network(layers)
    loss_fn = SoftmaxCEL()
    optimizer = optimizer_class(layers, **optimizer_kwargs)

    epoch_losses = []
    if isinstance(optimizer, SGD):
        print("SGD")
    elif isinstance(optimizer, SGDMomentum):
        print("SGDMomentum")
    elif isinstance(optimizer, Adam):
        print("Adam")

    for epoch in range(epochs):
        np.random.seed(epoch)
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        epoch_loss = 0

        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i: i + batch_size]
            y_batch = y_shuffled[i: i + batch_size]

            Z = network.forward(X_batch)
            loss = loss_fn.forward(Z, y_batch)
            epoch_loss += loss

            dX = loss_fn.backward()
            network.backward(dX)
            optimizer.step()

        epoch_losses.append(epoch_loss / (len(X_train) // batch_size))
        print(f"Epoch {epoch + 1}: loss {epoch_losses[-1]:.4f}")

    return epoch_losses


train_epochs = 30

sgd_losses = train(SGD, {'lr': 0.01}, epochs=train_epochs)
momentum_losses = train(SGDMomentum, {'lr': 0.01, 'beta': 0.9}, epochs=train_epochs)
adam_losses = train(Adam, {'lr': 0.001}, epochs=train_epochs)

plt.figure(figsize=(15, 5))
plt.plot(sgd_losses, label='SGD')
plt.plot(momentum_losses, label='SGD + Momentum')
plt.plot(adam_losses, label='Adam')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SGD vs Momentum vs Adam')
plt.ylim(0, 1)
plt.legend()
plt.savefig('comparison.png')
plt.show()
