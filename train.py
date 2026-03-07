import numpy as np

from tqdm import tqdm
from keras.datasets import mnist

from src.network import Network
from src.layers import Linear, ReLU
from src.loss import SoftmaxCEL
from src.optimizer import SGD

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype(np.float64) / 255.0
X_test = X_test.reshape(10000, 784).astype(np.float64) / 255.0

y_train = np.eye(10)[y_train]  # (60000, 10)
y_test = np.eye(10)[y_test]  # (10000, 10)

layers = [Linear(784, 128), ReLU(),
          Linear(128, 64), ReLU(),
          Linear(64, 10)]

network = Network(layers)
optimizer = SGD(layers, 0.01)
loss_fn = SoftmaxCEL()

Z = network.forward(X_train)
loss = loss_fn.forward(Z, y_train)


def accuracy(X, y_true):
    preds = np.argmax(network.forward(X), axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(preds == labels)


# ── Training Loop ─────────────────────────────────────
epochs = 20
batch_size = 32
n_batches = len(X_train) // batch_size

for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0

    with tqdm(range(0, len(X_train), batch_size),
              desc=f"Epoch {epoch + 1}/{epochs}",
              unit="batch") as pbar:

        for i in pbar:
            X_batch = X_shuffled[i: i + batch_size]
            y_batch = y_shuffled[i: i + batch_size]

            Z = network.forward(X_batch)
            loss = loss_fn.forward(Z, y_batch)
            epoch_loss += loss

            dX = loss_fn.backward()
            network.backward(dX)
            optimizer.step()

            pbar.set_postfix(loss=f"{epoch_loss / (i // batch_size + 1):.4f}")

    train_acc = accuracy(X_train, y_train)
    test_acc = accuracy(X_test, y_test)
    print(f"  train acc: {train_acc:.4f}  |  test acc: {test_acc:.4f}\n")

network.save('weights/model')
