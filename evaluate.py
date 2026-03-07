import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from src.network import Network
from src.layers import Linear, ReLU

(_, _), (X_test, y_test) = mnist.load_data()

X_test_flat = X_test.reshape(10000, 784).astype(np.float64) / 255.0
y_test_onehot = np.eye(10)[y_test]

layers = [Linear(784, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, 10)]
network = Network(layers)
network.load('weights/model')

Z = network.forward(X_test_flat)
preds = np.argmax(Z, axis=1)  # predicted class
labels = y_test  # true class (integers, not one-hot)

wrong_indices = np.where(preds != labels)[0]
print(f"Total errors: {len(wrong_indices)} / {len(y_test)}  "
      f"({100 * len(wrong_indices) / len(y_test):.2f}% error rate)")

# ── Per-class breakdown ───────────────────────────────
print("\nErrors per class:")
for digit in range(10):
    total_in_class = np.sum(labels == digit)
    errors_in_class = np.sum((labels == digit) & (preds != labels))
    print(f"  digit {digit}: {errors_in_class:3d} / {total_in_class} wrong  |  {(errors_in_class / total_in_class):.4f}% Error Rate")

# ── Most confused pairs ───────────────────────────────
print("\nMost common confusions (true → predicted):")
from collections import Counter

confusions = Counter(zip(labels[wrong_indices], preds[wrong_indices]))
for (true, pred), count in confusions.most_common(10):
    print(f"  {true} mistaken for {pred}: {count} times")

# ── Plot sample mistakes ──────────────────────────────
n_show = 16
sample = wrong_indices[:n_show]

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle("Misclassified Samples", fontsize=14)

for ax, idx in zip(axes.flat, sample):
    ax.imshow(X_test[idx], cmap='gray')
    ax.set_title(f"true: {labels[idx]}  pred: {preds[idx]}", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig("mistakes.png")
plt.show()
print("\nSaved plot to mistakes.png")
