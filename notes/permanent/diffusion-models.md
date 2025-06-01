---
title: Diffusion Models
date: 2024-10-03 00:00
modified: 2024-10-03 00:00
status: draft
---

**Diffusion Models** also known as **Diffusion Probabilitistic Models** are a type of generative model that learn to generate data by reversing a gradual, noise-driven diffusion process, inspired by [Nonequilibrium Thermodynamics](../../../permanent/nonequilibrium-thermodynamics.md).

First introduced in [Deep Unsupervised Learning Using Nonequilibrium Thermodynamics](../../../reference/deep-unsupervised-learning-using-nonequilibrium-thermodynamics.md), in which they describe the key steps involved in training diffusion models.

Forward Diffusion: Add [Gaussian Noise](gaussian-noise.md) to data over multiple steps.
Reverse Diffusion: Learn to reverse the process. Go from noise to data.