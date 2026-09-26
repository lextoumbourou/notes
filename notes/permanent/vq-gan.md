---
title: VQ-GAN
date: 2024-01-09 00:00
modified: 2026-09-26 08:50
status: draft
---

**VQ-GAN** is an image model from [Taming Transformers for High-Resolution Image Synthesis](taming-transformers-for-high-resolution-image-synthesis.md) that combines the discrete codebook of [VQ-VAE](vq-vae.md) with adversarial training from [Generative Adversarial Network](generative-adversarial-network.md).

A convolutional encoder compresses an image into a small grid of codebook indices, and the decoder is trained with a perceptual loss and a patch-based discriminator, so it can reconstruct sharp images from a highly compressed code. A [Transformer](transformer.md) is then trained autoregressively over the code indices to generate high-resolution images. See also [Improved VQGAN](improved-vqgan.md).
