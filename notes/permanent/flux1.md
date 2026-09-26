---
title: FLUX.1
date: 2025-06-01 00:00
modified: 2026-09-26 08:55
status: draft
---

FLUX.1 is a text-to-image flow-matching transformer.

A rectified flow transformer trained in the latent space of an image autoencoder.

train a convolutional autoencoder with an adversarial objective from scratch

By scaling up the training compute and using 16 latent channels, we improve the reconstruction capabilities compared to related models; see Section 2.

FLUX.1 is built from a mix of double stream and single stream blocks.

Double stream blocks employ separate weights for image and text tokens, and mixing is done by applying the attention operation over the concatenation of tokens. After passing the sequences through the double stream blocks, we discard the text tokens and apply 38 single stream blocks to the image tokens.

To improve GPU utilization of single stream blocks, we leverage fused feed-forward blocks inspired by Dehghani et al. [8], which i) reduce the number of modulation parameters in a feedforward block by a factor of 2 and ii) fuse the attention input- and output linear layers with that of the MLP, leading to larger matrix-vector multiplications and thus more efficient training and inference. We utilize factorised three–dimensional Rotary Positional Embeddings (3D RoPE) [53]. Every latent token is indexed by its space-time coordinates (t, h, w) (with t ≡ 0 for single image inputs). See Figure 3 for a visualisation
