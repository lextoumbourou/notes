---
title: Auto-Encoder
date: 2024-07-19 00:00
modified: 2026-09-26 08:55
status: draft
---

An **Auto-Encoder** is a neural network trained to reconstruct its own input. It has two parts: an encoder, which compresses the input into a lower-dimensional latent representation (the bottleneck), and a decoder, which reconstructs the input from that representation. See [Encoder-Decoder](encoder-decoder.md).

It's trained with a reconstruction loss, such as mean squared error between the input and output. Because the bottleneck is smaller than the input, the network is forced to learn a compressed representation that keeps the most important information.

Auto-encoders are used for [Dimensionality Reduction](dimensionality-reduction.md), denoising and representation learning. The [Variational Auto-Encoder](variational-auto-encoder.md) is a probabilistic version that can also be used to generate new samples.
