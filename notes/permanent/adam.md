---
title: Adam
date: 2026-01-19 00:00
modified: 2026-09-26 08:55
status: draft
---

**Adam** (Adaptive Moment Estimation) is an optimisation algorithm for training neural networks, introduced by Diederik Kingma and Jimmy Ba in 2014. It's an extension of [Stochastic Gradient Descent](stochastic-gradient-descent.md).

Adam keeps an exponential moving average of the gradients (the first moment) and of the squared gradients (the second moment) for each parameter. It uses these to give each parameter its own effective step size, and applies a bias correction because both averages start at zero.

The commonly used default hyperparameters are $\beta_1 = 0.9$, $\beta_2 = 0.999$ and $\epsilon = 10^{-8}$. It works well out of the box on a lot of problems, which is why it's one of the most popular optimisers in deep learning.
