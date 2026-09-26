---
title: Dropout Regularisation
date: 2026-09-26 08:52
modified: 2026-09-26 08:52
status: draft
---

**Dropout Regularisation** is a regularisation technique where, during training, each neuron's output is randomly set to zero with some probability $p$.

It stops the network from relying too heavily on any single neuron, which encourages it to learn more robust features and reduces overfitting. Dropout is switched off at inference time. To keep the expected activations the same in both cases, the remaining activations are usually scaled by $\frac{1}{1-p}$ during training.

It was described in the paper [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) by Srivastava et al. (2014).

The idea has inspired a number of other techniques. See [Dropout-based Techniques](dropout-based-techniques.md).
