import matplotlib.pyplot as plt
import numpy as np

from src.layers import Linear, ReLU
from src.optimizer import SGD, SGDMomentum, Adam
from train import train
from src.scheduler import ExponentialLR

EPOCHS = 20

configs = [
    ('SGD', SGD, {'lr': 0.01}, None),
    ('SGD + Scheduler', SGD, {'lr': 0.01}, ExponentialLR),
    ('Adam', Adam, {'lr': 0.001}, None),
    ('Adam + Scheduler', Adam, {'lr': 0.001}, ExponentialLR),
]

results = {}
for label, optimizer_class, optimizer_kwargs, scheduler_class in configs:
    print(f"\n── {label} ──────────────────────")
    np.random.seed(42)
    layers = [Linear(784, 128), ReLU(),
              Linear(128, 128), ReLU(),
              Linear(128, 64), ReLU(),
              Linear(64, 10)]
    optimizer = optimizer_class(layers, **optimizer_kwargs)
    scheduler = scheduler_class(optimizer, gamma=0.95) if scheduler_class else None

    _, losses, _, _ = train(layers, optimizer, epochs=EPOCHS, scheduler=scheduler)
    results[label] = losses

plt.figure(figsize=(15, 5))
for label, losses in results.items():
    plt.plot(losses, label=label)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SGD vs Momentum vs Adam')
plt.ylim(0, 1)
plt.legend()
plt.savefig('comparison_optimizers.png')
plt.show()
