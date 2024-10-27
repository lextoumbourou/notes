---
title: Were RNNs All We Needed?
date: 2024-10-07 00:00
modified: 2024-10-07 00:00
status: draft
tags:
- MachineLearning
- SequenceModels
---

## Overview

Papers revisits [Recurrent Neural Networks](../../../../permanent/recurrent-neural-networks.md), specifically [LSTM](../../permanent/lstm.md) and [Gated Recurrent Neural Networks](Gated%20Recurrent%20Neural%20Networks), they show that by removing their hidden state dependencies from their input, forget and update gates, LSTMs an GRUs no longer need [[Backpropagation Through Time]], meaning they can be trained in paralell.

They introduce minimal versions that [[minLSTM]] and [[minGRU]].

They use:
(1) use significantly fewer parameters than their traditional counterparts
(2) are fully parallelizable during training (175× faster for a sequence of length 512).

They show these stripped-down versions of decade-old RNNs match the empirical performance of recent sequence models.

## Related Papers

* [Long Short-Term Memory (1997)](long-short-term-memory-1997.md)
