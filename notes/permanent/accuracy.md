---
title: Accuracy
date: 2026-03-08 00:00
modified: 2026-09-26 08:55
status: draft
---

**Accuracy** is an evaluation metric for [Classification](classification.md) models: the proportion of predictions the model got right.

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

It's easy to understand, but it can be misleading on imbalanced datasets. If 99% of examples are negative, a model that always predicts negative gets 99% accuracy while being useless. In those cases, metrics like [Precision](precision.md), [Recall](recall.md) and the [F1-Score](f1-score.md), or looking at the full [Confusion Matrix](confusion-matrix.md), tell you more.
