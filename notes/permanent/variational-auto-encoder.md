---
title: Variational Auto-Encoder
date: 2024-01-09 00:00
modified: 2026-09-26 08:50
status: draft
---

A **Variational Auto-Encoder** (VAE) is a [Generative Model](generative-model.md) that extends the [Auto-Encoder](autoencoder.md) by learning a probability distribution over the latent space, instead of mapping each input to a single point.

The encoder outputs the parameters (usually the mean and variance of a Gaussian) of a distribution for each input, a latent vector is sampled from it using the reparameterisation trick, and the decoder reconstructs the input from that sample. It's trained with a reconstruction loss plus a KL divergence term that keeps the learned distributions close to a simple prior, typically a standard normal. Because the latent space ends up smooth, you can sample from the prior and decode it to generate new data. VAEs were introduced by Kingma and Welling in 2013, and [VQ-VAE](vq-vae.md) is a variant with discrete latents.
