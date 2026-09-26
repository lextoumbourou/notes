---
title: Recurrent Neural Networks
date: 2024-01-28 00:00
modified: 2026-09-26 08:55
status: draft
---

**Recurrent Neural Networks** (RNNs) are neural networks for sequential data that process one step at a time, passing a hidden state from each step to the next, so the network has a memory of what it's seen so far.

They're trained with [Backpropagation](backpropagation.md) through time, and plain RNNs struggle with long sequences because of vanishing gradients. Gated variants like the [LSTM](lstm.md) and [Gated Recurrent Unit](gated-recurrent-unit.md) help with this.

Because each step depends on the previous one, RNNs are hard to parallelise, which is one reason the [Transformer](transformer.md) replaced them for many tasks. See also [Were RNNs All We Needed?](../reference/papers/were-rnns-all-we-needed.md)
