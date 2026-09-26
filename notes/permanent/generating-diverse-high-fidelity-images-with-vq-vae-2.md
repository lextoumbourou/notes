---
title: Generating Diverse High-Fidelity Images with VQ-VAE-2
date: 2024-01-10 00:00
modified: 2026-09-26 08:50
status: draft
---

Notes from paper [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446) by Ali Razavi, Aaron van den Oord, Oriol Vinyals

**Generating Diverse High-Fidelity Images with VQ-VAE-2** is a 2019 paper that introduces **VQ-VAE-2**, a hierarchical version of [VQ-VAE](vq-vae.md) for image generation.

Instead of a single grid of discrete latent codes, VQ-VAE-2 uses a hierarchy: a small top-level code that captures global structure, and a larger bottom-level code that captures local detail. Autoregressive PixelCNN-style priors are then trained over the codes, and sampling from them and decoding gives new images. Because the priors work in the compressed latent space rather than on raw pixels, sampling is much faster, and the paper showed samples with quality competitive with GANs of the time, but with more diversity.
