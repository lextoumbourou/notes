---
title: Pooling Layer
date: 2026-05-04 00:00
modified: 2026-09-26 08:55
status: draft
---

**Pooling Layer** is a layer in a [Convolutional Neural Network](convolutional-neural-network.md) that downsamples feature maps by summarising small regions of them.

The most common type is max pooling, which takes the maximum value in each window. For example, a 2x2 window with a stride of 2 halves the width and height of the feature map. Average pooling takes the mean of each window instead.

Pooling reduces the amount of computation in later layers, and makes the network a bit more robust to small shifts in the input. Unlike a [Convolutional Layer](convolutional-layer.md), it has no learnable parameters (see [Kernel](kernel.md)).
