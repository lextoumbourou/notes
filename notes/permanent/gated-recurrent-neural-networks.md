---
title: Gated Recurrent Neural Networks
date: 2026-09-26 08:55
modified: 2026-09-26 08:55
status: draft
---

**Gated Recurrent Neural Networks** are [Recurrent Neural Networks](recurrent-neural-networks.md) that use learned gates to control how information flows through the hidden state: what to keep, what to forget and what to output at each step.

The two best-known examples are the [LSTM](lstm.md) and the [Gated Recurrent Unit](gated-recurrent-unit.md) (GRU). The gates help with the vanishing gradient problem that makes plain RNNs struggle to learn long-range dependencies.

Before [Attention Is All You Need](attention-is-all-you-need.md), gated RNNs were the go-to models for sequence modelling tasks like [Machine Translation](machine-translation.md). See also [Were RNNs All We Needed?](../reference/papers/were-rnns-all-we-needed.md).
