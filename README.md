# Neural Network from Scratch — MNIST & Fashion-MNIST

A fully hand-rolled neural network classifier built with **pure Python and NumPy** — no PyTorch, no TensorFlow, no autograd. Every forward pass, every backpropagation step, every optimizer update, every normalization layer is implemented from first principles.

**Achieved ~97% accuracy on MNIST and ~88% on Fashion-MNIST.**

---

## What's Implemented

### Layers
- **`Linear`** — forward pass `X @ W + b`, full backward pass with `dW`, `db`, `dX`. Xavier initialization.
- **`ReLU`** — forward and backward with boolean gradient mask `(X > 0) * dX`
- **`Dropout`** — inverted dropout with configurable drop probability `p`. Scales surviving neurons by `1/(1-p)` during training, passes through unchanged at inference
- **`BatchNorm`** — full batch normalization with learnable `γ` and `β`, running mean/variance tracking for inference, complete backward pass through all three gradient paths

### Loss
- **`SoftmaxCEL`** — Softmax and Cross-Entropy Loss fused into a single layer. Uses the combined gradient simplification `(ŷ - y) / N`, numerically stable with `log(clip(ŷ, 1e-9, 1.0))`

### Optimizers
- **`SGD`** — vanilla stochastic gradient descent
- **`SGDMomentum`** — velocity accumulation, standard PyTorch-style formula `v = β·v + dW`
- **`Adam`** — adaptive moment estimation with first moment `m`, second moment `v`, and bias correction `m̂`, `v̂`. Updates both `Linear` weights and `BatchNorm` γ/β

### Scheduler
- **`ExponentialLR`** — multiplies optimizer `lr` by `gamma` after each epoch. `gamma=0.95` recommended starting point

### Network
- **`Network`** — chains layers for forward and backward pass. Save/load weights including `BatchNorm` running statistics via `np.savez`

---

## Project Structure

```
MNIST/
├── src/
│   ├── network.py          # Network — forward, backward, save, load
│   ├── layers.py           # Linear, ReLU, Dropout, BatchNorm, base Layer
│   ├── loss.py             # SoftmaxCEL — fused Softmax + Cross-Entropy
│   ├── optimizer.py        # SGD, SGDMomentum, Adam
│   ├── scheduler.py        # ExponentialLR
│   └── data_loader.py      # MNIST / Fashion-MNIST loading and preprocessing
├── experiments/
│   ├── compare_optimizers.py   # SGD vs Momentum vs Adam loss curves
│   └── compare_dropout.py      # Dropout effect on train/test accuracy gap
├── weights/
│   └── model.npz               # saved weights + BatchNorm running stats
├── train.py                    # shared train() function + main entry point
├── evaluate.py                 # error analysis and misclassification plots
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/yourusername/MNIST
cd MNIST

python -m venv mnist-env
source mnist-env/bin/activate      # Windows: mnist-env\Scripts\activate

pip install -r requirements.txt

python train.py
python evaluate.py
python experiments/compare_optimizers.py
python experiments/compare_dropout.py
```

---

## The `train()` Function

All training runs share a single function in `train.py`:

```python
network, losses, train_accs, test_accs = train(
    layers,
    optimizer,
    epochs=15,
    batch_size=32,
    scheduler=scheduler   # optional — pass None to skip
)
```

It handles Dropout and BatchNorm mode switching automatically, prints loss/accuracy/gap per epoch, and returns all metrics for plotting.

---

## Optimizer Comparison

| Optimizer | Final Loss (15 epochs) | Notes |
|---|---|---|
| SGD | ~0.30 | baseline, slowest convergence |
| SGD + Momentum | ~0.22 | smoother curve, better minimum |
| Adam | ~0.20 | fastest adaptation, best final loss |

---

## Dropout Effect

Dropout's effect is only visible when the network is large enough to overfit. The comparison uses a deliberately oversized architecture (`784→512→512→256→256→10`) where the train/test gap is clearly visible without dropout and closes significantly with it.

| | Train Acc | Test Acc | Gap |
|---|---|---|---|
| No Dropout | ~0.95 | ~0.88 | ~0.07 |
| Dropout p=0.3 | ~0.91 | ~0.89 | ~0.02 |

---

## The Math

Everything derived from first principles:

**Linear backward:**
```
dW = X.T @ dX
db = sum(dX, axis=0)
dX = dX @ W.T
```

**ReLU backward:**
```
dX = (X > 0) * dX
```

**Combined Softmax + CEL gradient** — derived through the full 10×10 Jacobian, simplifies to:
```
dZ = (ŷ - y) / N
```

**Adam bias correction** — corrects zero-initialization of `m` and `v`:
```
m̂ = m / (1 - β1^t)
v̂ = v / (1 - β2^t)
```

**BatchNorm backward** — three separate gradient paths through the computation graph:
```
dX = dX̂/√(σ²+ε) + dσ²·2(X-μ)/N + dμ/N
```

**Inverted Dropout scaling** — surviving neurons amplified by exactly the inverse of their survival probability:
```
output = X * mask * 1/(1-p)
```

---

## Switching Datasets

One line in `src/data_loader.py`:

```python
from keras.datasets import fashion_mnist   # or mnist
dataset = fashion_mnist
```

No other changes needed — both datasets are 60,000 × 28×28 grayscale images, 10 classes.

---

## Using BatchNorm

```python
from src.layers import Linear, ReLU, BatchNorm

layers = [
    Linear(784, 128), BatchNorm(128), ReLU(),
    Linear(128, 64),  BatchNorm(64),  ReLU(),
    Linear(64, 10)    # no BatchNorm on final layer
]
```

BatchNorm goes after Linear, before activation. The training/inference flag is toggled automatically by `train()`.

---

## Requirements

```
numpy
keras
tqdm
matplotlib
```