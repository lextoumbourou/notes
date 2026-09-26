---
title: Importance Sampling
date: 2026-09-26 08:55
modified: 2026-09-26 08:55
status: draft
---

**Importance Sampling** is a technique for estimating an expectation under a target distribution $p(x)$ that is hard to sample from, by drawing samples from an easier proposal distribution $q(x)$ and weighting each sample by $\frac{p(x)}{q(x)}$:

$$
\mathbb{E}_{p}[f(x)] = \mathbb{E}_{q}\left[f(x)\frac{p(x)}{q(x)}\right] \approx \frac{1}{N}\sum_{i=1}^{N} f(x_i)\frac{p(x_i)}{q(x_i)}, \quad x_i \sim q
$$

It works well when $q$ is close to $p$. When they're very different, especially in high dimensions, a few samples end up with huge weights and the estimate has high variance. [Annealed Importance Sampling](annealed-importance-sampling.md) addresses this by moving gradually from $q$ to $p$ through a sequence of intermediate distributions.
