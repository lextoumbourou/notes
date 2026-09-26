---
title: Gaussian Mixture Model
date: 2024-04-11 00:00
modified: 2026-09-26 08:50
status: draft
---

A **Gaussian Mixture Model** (GMM) is a special case of [Mixture Model](mixture-model.md) where each component is a Gaussian (normal) distribution.

The density is a weighted sum of $K$ Gaussians:

$$
p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

Where the mixing weights $\pi_k$ are non-negative and sum to 1, and each component has its own mean $\mu_k$ and covariance $\Sigma_k$. The parameters are usually fitted with the Expectation-Maximisation (EM) algorithm. GMMs are commonly used for soft clustering, where each point gets a probability of belonging to each cluster, and for density estimation.
