---
title: Sliding Window Attention
date: 2024-03-18 00:00
modified: 2024-03-18 00:00
status: draft
---

**Sliding Window Attention** is a modification to [Attention Mechanism](attention-mechanism.md), which modifies the causal mask, so tokens can only attend to $W$ past tokens. Here, we can see an example where $W = 3$

![](../../../_media/sliding-window-attention-sliding-window.png)
*Image from [mistral-src](https://github.com/mistralai/mistral-src?tab=readme-ov-file).*

Tokens outside the sliding window can still influence next-word prediction, and information can move forward by W tokens in each attention layer. For example, after four attention layers with a sliding window of 4k tokens, information can propagate to a context length of 16k.

![](../../../_media/sliding-window-attention-propogation.png)
*Image from [mistral-src](https://github.com/mistralai/mistral-src?tab=readme-ov-file).*

References:
* [Mistral 7B](https://arxiv.org/abs/2310.06825)
* [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf)
* [Generating long sequences with sparse transformers](https://arxiv.org/abs/1904.10509).
