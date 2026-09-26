---
title: Kernel
date: 2026-05-04 00:00
modified: 2026-09-26 08:55
status: draft
---

A **Kernel** (also called a filter), in a [Convolutional Neural Network](convolutional-neural-network.md), is a small matrix of learnable weights that slides across the input, computing a weighted sum at each position to produce a feature map.

For example, a 3x3 kernel looks at a 3x3 patch of pixels at a time. Kernels in early layers tend to learn simple features like edges, and deeper layers combine them into more complex patterns.

See [Convolutional Layer](convolutional-layer.md) and [Pooling Layer](pooling-layer.md).
