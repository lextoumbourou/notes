---
title: "Sana - Efficient High-Resolution Image Synthesis With Linear Diffusion Transformers"
date: 2026-05-24 00:00
modified: 2026-09-26 08:55
status: draft
---

Paper from NVIDIA that introduces [Sana](sana.md), a text-to-image model that can efficiently generate high-resolution images, up to 4096 × 4096.

Genesis of [Linear Diffusion Transformer](linear-diffusion-transformer.md), which replace the standard attention in a Diffusion Transformer with [Linear Attention](linear-attention.md), so compute grows linearly with the number of tokens instead of quadratically.

It pairs this with a deep compression autoencoder that compresses images 32× (instead of the usual 8×), which means far fewer tokens for the transformer to process.
