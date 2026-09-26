---
title: Layer Norm
date: 2024-04-07 00:00
modified: 2026-09-26 09:08
status: draft
---

**Layer Norm** is an alternative to **Batch Norm** which fixes 3 issues with Batch Norm:
- Hard to use with sequence data due to sequences of varying length.
- Hard to use with small batch sizes.
- Hard to parallelise.

Layer norm calculates the normalisation statistics across the features in a layer, instead of across the batch.

Input values in all neurons in the same layer are normalised for each data sample.

Normalisation rescales numbers into a standard range. The standard way to do it is to subtract the mean and divide by the standard deviation, which gives values with a mean of 0 and a standard deviation of 1. Layer norm then applies a learnable scale ($\gamma$) and shift ($\beta$):

$$
y = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Coding it:

```python
import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta
```
