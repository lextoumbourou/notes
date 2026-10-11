---
title: Scaled Dot-Product Attention
date: 2024-03-13 00:00
modified: 2026-05-25 00:00
summary: The specific self-attention formulation from the Transformer paper, distinguished by scaling scores by the square root of the attention dimension.
cover: /_media/scaled-dot-product-attention.png
tags:
- MachineLearning
- LargeLanguageModels
category: note
---

Scaled Dot-Product Attention is the specific formulation of [Self-Attention](self-attention.md) introduced in [Attention Is All You Need](attention-is-all-you-need.md) [@vaswaniAttentionAllYou2017]. It is used in the [Transformer](transformer.md) architecture and in most subsequent large language models.

The mechanism is identical to standard dot-product self-attention, with one addition: scores are divided by the square root of the attention dimension before the softmax ($\color{red}{\sqrt{d_k}}$).

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\color{red}{\sqrt{d_k}}}\right)V$$

## Why scale?

As the attention dimension $d_k$ grows, the dot products between query and key vectors tend to grow in magnitude: there are more terms being summed. Large values fed into the softmax push it into regions with very small gradients, making training slow and unstable.

Dividing by $\sqrt{d_k}$ keeps the scores in a stable range without changing their relative ordering, since all scores are scaled by the same factor [@vaswaniAttentionAllYou2017].

```python {run=false}
scores = query @ key.transpose(2, 1)
scores = scores / math.sqrt(attention_dim)  # the "scaled" part
scores = softmax(scores, dim=-1)
out = scores @ value
```

## Why $\sqrt{d_k}$?

Well, because it's the [Standard Deviation](standard-deviation.md) of the [Dot Product](dot-product.md).

- If you suppose the components of $q$ and $k$ are independent, each with mean 0 and variance 1 (standardise [Normal Distribution](normal-distribution.md))
- Then q*k = \sum(qiki) os the sum of d_k terms
- So the sum has mean 0 and variance d_k, which means a standard deviation of √d_k.
- Dividing by by √d_k brings the variance back to 1 whatever the head size is. The softmax then sees inputs of the same scale at d_k = 64 or d_k = 1024.
- Dividing by d_k would shrink the scores too much and flatten the attention towards uniform. Dividing by √d_k is the exact correction for how the spread grows.


---

For a full walkthrough of the self-attention mechanism including input preparation, the QKV projections, masking, and the complete PyTorch module, see [Self-Attention](self-attention.md).
