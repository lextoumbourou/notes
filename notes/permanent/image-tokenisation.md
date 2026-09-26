---
title: Image Tokenisation
date: 2026-02-02 00:00
modified: 2026-09-26 09:05
status: draft
---

**Image Tokenisation** is the process of converting an image into a sequence of tokens that a [Transformer](transformer.md) can process, similar to [Tokenisation](tokenisation.md) for text.

There are two common approaches:

* Split the image into fixed-size patches and project each patch into an embedding. These are continuous tokens.
* Use a learned encoder with a [Codebook](codebook.md), like [VQ-VAE](vq-vae.md) or [VQ-GAN](vq-gan.md), to map regions of the image to discrete token IDs. This lets image generation be treated like language modelling.

See also [Audio Tokenisation](audio-tokenization.md).
