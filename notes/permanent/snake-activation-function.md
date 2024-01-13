---
title: Snake Activation Function
date: 2024-01-04 00:00
modified: 2024-01-04 00:00
status: draft
---

An [Activation Function](activation-function.md) function from paper [Neural networks fail to learn periodic functions and how to fix it](../../../permanent/neural-networks-fail-to-learn-periodic-functions-and-how-to-fix-it.md).

Useful when a periodic induction bias is required, as described in [High-Fidelity Audio Compression with Improved RVQGAN](../reference/papers/high-fidelity-audio-compression-with-improved-rvqgan.md).

Function:

$\text{snake}(x) = x + \frac{1}{\alpha} \sin^2(\alpha)$
