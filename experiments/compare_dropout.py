import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.layers import Linear, ReLU, Dropout
from src.network import Network
from src.loss import SoftmaxCEL
from src.optimizer import Adam
from src.data_loader import X_train, y_train, X_test, y_test


def accuracy(network, X, y_true):
    preds = np.argmax(network.forward(X), axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(preds == labels)


def train(use_dropout, dropout_p=0.3, epochs=15, batch_size=32):
    np.random.seed(42)

    if use_dropout:
        # layers = [
        #     Linear(784, 256), ReLU(), Dropout(p=dropout_p),
        #     Linear(256, 128), ReLU(), Dropout(p=dropout_p),
        #     Linear(128, 128), ReLU(), Dropout(p=dropout_p),
        #     Linear(128, 64), ReLU(), Dropout(p=dropout_p),
        #     Linear(64, 10)]
        layers = [
            Linear(784, 512), ReLU(), Dropout(p=dropout_p),
            Linear(512, 512), ReLU(), Dropout(p=dropout_p),
            Linear(512, 256), ReLU(), Dropout(p=dropout_p),
            Linear(256, 256), ReLU(), Dropout(p=dropout_p),
            Linear(256, 10)]

    else:
        # layers = [
        #     Linear(784, 128), ReLU(),
        #     Linear(128, 128), ReLU(),
        #     Linear(128, 64), ReLU(),
        #     Linear(64, 10)]
        layers = [
            Linear(784, 512), ReLU(),
            Linear(512, 512), ReLU(),
            Linear(512, 256), ReLU(),
            Linear(256, 256), ReLU(),
            Linear(256, 10)]

    network = Network(layers)
    loss_fn = SoftmaxCEL()
    optimizer = Adam(layers, lr=0.001)

    label = f"Dropout p={dropout_p}" if use_dropout else "No Dropout"
    print(f"\n── {label} ──────────────────────")

    epoch_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # training mode
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

        # switch to inference mode for evaluation
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
              f"gap: {train_acc - test_acc:.4f}")  # overfitting gap printed directly

    return epoch_losses, train_accuracies, test_accuracies


# ── Run ───────────────────────────────────────────────
epochs = 15

dr = 0.3
no_dropout_losses, no_dropout_train, no_dropout_test = train(use_dropout=False, epochs=epochs)
dropout_losses, dropout_train, dropout_test = train(use_dropout=True, dropout_p=dr, epochs=epochs)

# ── Plot ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(no_dropout_losses, label='No Dropout')
axes[0].plot(dropout_losses, label=f'Dropout p={dr}')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].set_ylim(0, 1)
axes[0].legend()

axes[1].plot(no_dropout_train, label='No Dropout — train', linestyle='--')
axes[1].plot(no_dropout_test, label='No Dropout — test')
axes[1].plot(dropout_train, label='Dropout — train', linestyle='--')
axes[1].plot(dropout_test, label='Dropout — test')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Train vs Test Accuracy')
axes[1].set_ylim(0, 1)
axes[1].legend()

plt.suptitle('Dropout Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('comparison_dropout.png')
plt.show()
