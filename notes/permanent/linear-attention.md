---
title: "Linear Attention"
date: 2026-05-25 00:00
modified: 2026-09-26 09:05
status: draft
---

**Linear Attention** is a variant of the [Attention Mechanism](attention-mechanism.md) that reduces the cost of attention from quadratic to linear in the sequence length.

Standard [Scaled-Dot Product Attention](scaled-dot-product-attention.md) computes $\text{softmax}(QK^T)V$, which needs an $N \times N$ attention matrix for $N$ tokens. Linear attention replaces the softmax with a kernel feature map $\phi$, so attention becomes $\phi(Q)(\phi(K)^T V)$. Because matrix multiplication is associative, $\phi(K)^T V$ can be computed first, which avoids ever building the $N \times N$ matrix.

The trade-off is that it's usually less expressive than softmax attention. It's used in the [Linear Diffusion Transformer](linear-diffusion-transformer.md) from [Sana](sana.md).
