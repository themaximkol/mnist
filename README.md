# Neural Network from Scratch — MNIST & Fashion-MNIST

A fully hand-rolled neural network classifier built with **pure Python and NumPy** — no PyTorch, no TensorFlow, no autograd. Every forward pass, every backpropagation step, every optimizer update is implemented from first principles.

---

## What This Is

This project is a ground-up implementation of a multi-layer perceptron (MLP) trained on MNIST and Fashion-MNIST. The goal was not to achieve state-of-the-art accuracy but to deeply understand what actually happens inside a neural network — by deriving and implementing every piece of the math by hand.

**Achieved ~97% accuracy on MNIST and ~88% on Fashion-MNIST.**

---

## What's Implemented

### Core Architecture
- `Linear` layer — forward pass `X @ W + b`, full backward pass with `dW`, `db`, `dX`
- `ReLU` activation — forward and backward with boolean gradient mask
- `SoftmaxCEL` — Softmax and Cross-Entropy Loss fused into a single layer using the combined gradient simplification `ŷ - y`

### Optimizers
- **SGD** — vanilla stochastic gradient descent
- **SGD with Momentum** — velocity accumulation with configurable `β`
- **Adam** — adaptive moment estimation with bias correction, `m̂` and `v̂`

### Training Infrastructure
- Mini-batch training loop with `tqdm` progress bars
- Per-epoch train/test accuracy tracking
- Weight serialization and loading via `np.savez`
- Optimizer comparison experiments with reproducible seeding

### Evaluation
- Per-class error breakdown
- Most confused digit pairs
- Visual grid of misclassified samples

---

## Project Structure

```
MNIST/
├── src/
│   ├── network.py          # Network class — forward, backward, save, load
│   ├── layers.py           # Linear, ReLU, base Layer class
│   ├── loss.py             # SoftmaxCEL — fused Softmax + Cross-Entropy
│   ├── optimizer.py        # SGD, SGDMomentum, Adam
│   └── data_loader.py      # MNIST / Fashion-MNIST loading and preprocessing
├── experiments/
│   └── compare_optimizers.py   # SGD vs Momentum vs Adam loss curves
├── weights/
│   └── model.npz               # saved weights
├── train.py                    # main training entry point
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
```

---

## Optimizer Comparison

Running `compare_optimizers.py` trains three identical networks from the same random seed and plots their loss curves side by side.

| Optimizer | Final Loss | Notes |
|---|---|---|
| SGD | ~0.30 | baseline, slowest convergence |
| SGD + Momentum | ~0.22 | smoother curve, better minimum |
| Adam | ~0.20 | fastest adaptation, best final loss |

---

## The Math

The backward pass is derived entirely from first principles:

- **Linear layer gradients** — `dW = X.T @ dX`, `db = sum(dX, axis=0)`, `dX = dX @ W.T`
- **ReLU gradient** — `(X > 0) * dX` — boolean mask passes gradient where input was positive
- **Combined Softmax + CEL gradient** — derivation through the full Jacobian shows the gradient simplifies to `(ŷ - y) / N`
- **Adam bias correction** — corrects for zero-initialization of moment estimates using `1 - β^t`

---

## Requirements

```
numpy
keras
tqdm
matplotlib
```

---

## Switching Datasets

Change one line in `src/data_loader.py`:

```python
from keras.datasets import fashion_mnist   # or mnist
```

No other changes needed — both datasets share identical shapes and preprocessing.
