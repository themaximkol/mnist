import numpy as np
from tqdm import tqdm

from src.network import Network
from src.loss import SoftmaxCEL
from src.data_loader import X_train, y_train, X_test, y_test
from src.layers import Linear, ReLU, Dropout
from src.optimizer import Adam
from src.scheduler import ExponentialLR


def accuracy(network, X, y_true):
    preds = np.argmax(network.forward(X), axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(preds == labels)


def train(layers, optimizer, epochs=15, batch_size=32, seed=42, scheduler=None):
    np.random.seed(seed)

    network = Network(layers)
    loss_fn = SoftmaxCEL()

    epoch_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        for layer in layers:
            if isinstance(layer, Dropout):
                layer.training = True

        np.random.seed(epoch)
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

        # step scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

        for layer in layers:
            if isinstance(layer, Dropout):
                layer.training = False

        train_acc = accuracy(network, X_train, y_train)
        test_acc = accuracy(network, X_test, y_test)

        epoch_losses.append(epoch_loss / (len(X_train) // batch_size))
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"  loss: {epoch_losses[-1]:.4f}  "
              f"train acc: {train_acc:.4f}  "
              f"test acc: {test_acc:.4f}  "
              f"gap: {train_acc - test_acc:.4f}  "
              f"lr: {optimizer.lr:.6f}")  # print current lr so you can see it decaying

    return network, epoch_losses, train_accuracies, test_accuracies


if __name__ == '__main__':
    layers = [Linear(784, 128), ReLU(),
              Linear(128, 64), ReLU(),
              Linear(64, 10)]

    optimizer = Adam(layers, lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    network, losses, train_accs, test_accs = train(
        layers, optimizer, epochs=20, scheduler=scheduler
    )
    network.save('weights/model')
