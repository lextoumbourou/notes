# CIFAR-10 classifier

A small convolutional neural network trained from random initialization (no pretrained weights) on CIFAR-10. The script downloads the dataset into `data/` if needed, trains with random crops and horizontal flips, then evaluates once on CIFAR-10's official 10,000-image test split.

## Result

5 training epochs on CPU; held-out test accuracy **50.19% (5,019 / 10,000)**. This is a short, from-scratch baseline, not a fully converged model. `results.json` includes example predictions (with true labels and confidence) for the first 12 held-out test images.

| Test image | Prediction | Actual | Confidence |
|---|---|---|---:|
| 0 | cat | cat | 20.0% |
| 1 | ship | ship | 49.8% |
| 2 | ship | ship | 48.5% |
| 3 | ship | airplane | 60.8% |
| 4 | frog | frog | 69.2% |
| 5 | frog | frog | 49.5% |
| 6 | truck | automobile | 42.7% |
| 7 | deer | frog | 30.4% |
| 8 | dog | cat | 38.9% |
| 9 | automobile | automobile | 53.5% |
| 10 | airplane | airplane | 48.3% |
| 11 | automobile | truck | 52.5% |

These are CIFAR-10 test images, not images used to train the network.

## Files

- `train.py` — model definition, training, evaluation, and sample prediction output.
- `cifar10_model.pt` — trained PyTorch checkpoint.
- `results.json` — measured score and held-out sample predictions.
- `data/` — downloaded CIFAR-10 dataset.

## Run

```sh
.venv/bin/python train.py --epochs 5
```

Requires Python, PyTorch, torchvision, and NumPy (installed in `.venv` in this workspace). Change `--epochs` to train longer; each run downloads the dataset only if absent.
