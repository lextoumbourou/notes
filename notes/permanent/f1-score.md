---
title: F1-Score
date: 2024-04-02 00:00
modified: 2024-04-02 00:00
status: draft
---

The F1-Score is an evaluation metric for binary classification problems, where 1 is a perfect score. It is the harmonic mean of precision and recall.

Precision is a measure of accuracy and is formulated as: $\frac{\text{TP}}{\text{TP} + \text{FP}}$

Recall is a measure of the true-positive rate and is formulated as: $\frac{\text{TP}}{\text{TP} + \text{FN}}$

Then, F1-score is the harmonic average of precision and recall: $\frac{2 \times \text{precision} \times \text{recall}}{(\text{precision} + \text{recall})}$

The choice of harmonic mean instead of arithmetic mean means that the model gives more weight to lower values, penalising models with highly imbalanced precision and recall scores.