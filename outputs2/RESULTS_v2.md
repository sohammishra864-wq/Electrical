# Version 2: self-contained simple RNN with a learning-rate experiment

Script: `rnn_load_classifier_v2.py`. Outputs: this folder (`outputs2/`). Version 1 (`outputs/`) is unchanged.

## What changed from version 1

| | Version 1 | Version 2 |
|---|---|---|
| RNN | PyTorch library (RNN / LSTM / GRU) | **Simple RNN written from scratch in numpy** |
| Training | Adam optimizer, lr 0.001 | **Plain mini-batch gradient descent**, lr swept from 0.25 to 0.55 |
| Gradients | computed automatically by the library | **hand-derived backpropagation through time**, checked against numerical slopes on every run |
| Split / scaling / metrics | scikit-learn | written out (about 10 lines each) |
| Circuit transient | SciPy ODE solver | **Euler method** loop (gives the same curves as the SciPy version) |
| Libraries | numpy, pandas, torch, sklearn, scipy, matplotlib, openpyxl | **numpy** (maths), **matplotlib** (graphs), **openpyxl** (Excel files) |

## The model

```
h_0 = 0
h_t = tanh(x_t Wx + h_(t-1) Wh + bh)      t = 1..4, one step per sweep level (90, 95, 100, 105 % V)
s   = h_4 Wy + by                         one score per class
p   = softmax(s)                          probabilities of CP, CC, CZ
loss = -ln p[true class]                  cross-entropy
update: W <- W - learning_rate x dLoss/dW (batch of 64 loads)
```

x_t = [V, I, P, V/V100, I/I100, P/P100] at sweep step t, with 32 hidden units, 60 epochs, and the weights from the best validation epoch kept.

## Learning-rate experiment (3 runs per learning rate, different random start weights)

| Learning rate | Validation acc. (mean +/- std) | Test acc. (mean +/- std) | Test error |
|---|---|---|---|
| 0.25 | 99.15 +/- 0.08 % | 99.14 +/- 0.07 % | 0.86 % |
| 0.30 | 99.15 +/- 0.05 % | 99.18 +/- 0.10 % | 0.82 % |
| 0.35 | 99.10 +/- 0.07 % | 99.17 +/- 0.06 % | 0.83 % |
| 0.40 | 99.14 +/- 0.05 % | 99.17 +/- 0.06 % | 0.83 % |
| 0.45 | 99.11 +/- 0.04 % | 99.19 +/- 0.02 % | 0.81 % |
| 0.50 | 99.11 +/- 0.07 % | 99.13 +/- 0.09 % | 0.87 % |
| 0.55 | 99.14 +/- 0.12 % | 99.14 +/- 0.05 % | 0.86 % |

**What it shows:** across 0.25 to 0.55, accuracy changes by less than 0.1 %. That is smaller than the run-to-run spread from
the random start weights alone. **Within this range the learning rate makes no measurable difference.** The network
reaches its best validation loss within about 6 to 11 epochs at every learning rate.

**Over-fitting is visible.** After about epoch 10, training loss keeps falling but validation loss slowly rises (graphs 09 and 13).
That is why the weights from the best validation epoch are kept rather than the weights from the last epoch.

The selection rule (highest mean validation accuracy) picked **0.25**. It tied exactly with 0.30 and won only because it comes
first in the list. Any value in the range would give the same result to within about 2 test loads.

## Final result: selected model (lr 0.25, seed 3) on the unseen test set (2,520 loads)

| Metric | Value |
|---|---|
| **Accuracy** | **99.21 %** |
| Precision (macro) | 99.21 % |
| Recall (macro) | 99.21 % |
| F1-score (macro) | 99.21 % |
| **Classification error** | **0.79 %** |

| | Pred CP | Pred CC | Pred CZ |
|---|---|---|---|
| **CP** | 837 | 3 | 0 |
| **CC** | 3 | 828 | 9 |
| **CZ** | 0 | 5 | 835 |

- All 20 errors involve CC, which sits between CP and CZ. CP and CZ are never confused with each other.
- Normal condition: 99.76 % (3 errors). Tough condition: 98.65 % (17 errors). Measurement noise is the main source of error.
- The from-scratch simple RNN does slightly better than version 1's library LSTM (99.21 % vs 98.85 %). The difference is
  about 9 test loads, so treat the two as equivalent.
