import matplotlib.pyplot as plt
import numpy as np

from src.layers import Linear, ReLU, Dropout
from src.optimizer import Adam
from train import train

EPOCHS = 15
DROPOUT_P = 0.3

configs = {
    'No Dropout': [
        Linear(784, 512), ReLU(),
        Linear(512, 512), ReLU(),
        Linear(512, 256), ReLU(),
        Linear(256, 256), ReLU(),
        Linear(256, 10)],

    f'Dropout p={DROPOUT_P}': [
        Linear(784, 512), ReLU(), Dropout(p=DROPOUT_P),
        Linear(512, 512), ReLU(), Dropout(p=DROPOUT_P),
        Linear(512, 256), ReLU(), Dropout(p=DROPOUT_P),
        Linear(256, 256), ReLU(), Dropout(p=DROPOUT_P),
        Linear(256, 10)],
}

results = {}
for label, layers in configs.items():
    print(f"\n── {label} ──────────────────────")
    np.random.seed(42)
    optimizer = Adam(layers, lr=0.001)
    _, losses, train_accs, test_accs = train(layers, optimizer, epochs=EPOCHS)
    results[label] = (losses, train_accs, test_accs)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for label, (losses, _, _) in results.items():
    axes[0].plot(losses, label=label)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].set_ylim(0, 1)
axes[0].legend()

for label, (_, train_accs, test_accs) in results.items():
    axes[1].plot(train_accs, label=f'{label} — train', linestyle='--')
    axes[1].plot(test_accs, label=f'{label} — test')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Train vs Test Accuracy')
axes[1].set_ylim(0, 1)
axes[1].legend()

plt.suptitle('Dropout Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('comparison_dropout.png')
plt.show()
