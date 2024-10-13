---
title: Hierarchical Model
date: 2024-02-03 00:00
modified: 2024-02-03 00:00
status: draft
---

The concept of hierarchical models in Deep Learning refers to a configuration where models feed output into another in sequence. In general, neural networks are said to be hierarchical architectures, since earlier layers feed outputs into future layers. However, some specific types of hierarchical models exist.

For example, the [VALL-E](vall-e.md) model took a hierarchical approach to modelling audio codes, by predicting later codes from earlier quantisers (see [Residual Vector Quantisation](residual-vector-quantization.md)).
