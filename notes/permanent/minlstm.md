---
title: minLSTM
date: 2026-09-26 08:55
modified: 2026-09-26 08:55
status: draft
---

**minLSTM** is a minimal version of the [LSTM](lstm.md) introduced in [Were RNNs All We Needed?](../reference/papers/were-rnns-all-we-needed.md).

Its forget, input and candidate hidden state only depend on the current input, not on the previous hidden state. That removes the need for backpropagation through time, so it can be trained in parallel using a parallel scan. It also uses significantly fewer parameters than a traditional LSTM.

See also [minGRU](mingru.md).
