---
category: note
title: Confusion Matrix
date: 2024-04-08 00:00
modified: 2026-07-24 20:47
status: draft
tags:
  - MachineLearning
  - Metrics
summary: A table of actual and predicted classes used to understand classification errors and calculate precision, recall and F1.
---

A **confusion matrix** counts how often a classifier predicts each class for examples whose actual classes are known. It shows not only how often a model is correct, but which classes it confuses.

## Binary confusion matrix

For binary classification, scikit-learn uses actual classes as rows and predicted classes as columns:

| | Predicted positive | Predicted negative |
| --- | ---: | ---: |
| **Actual positive** | True positive (TP) | False negative (FN) |
| **Actual negative** | False positive (FP) | True negative (TN) |

Some sources transpose these axes, so always read the labels rather than assuming an orientation.

- **True positive:** The example is positive and the model predicts positive.
- **False positive:** The example is negative but the model predicts positive.
- **False negative:** The example is positive but the model predicts negative.
- **True negative:** The example is negative and the model predicts negative.

In a binary hypothesis-testing interpretation, a false positive corresponds to a **Type I error**, while a false negative corresponds to a **Type II error**.

This meme has helped me remember Type I and Type II errors more than any other technique.

![Type I and Type II error meme](/_media/confusion-matrix-type-i-and-type-ii.png)

## Worked example

Suppose a model predicts whether someone will default:

| Actual | Predicted |
| ---: | ---: |
| 1 | 0 |
| 1 | 0 |
| 0 | 0 |
| 1 | 1 |

The confusion matrix is:

| | Predicted positive | Predicted negative |
| --- | ---: | ---: |
| **Actual positive** | 1 | 2 |
| **Actual negative** | 0 | 1 |

Therefore, $TP=1$, $FP=0$, $FN=2$ and $TN=1$.

## Precision

[Precision](../../../permanent/precision.md) asks: of everything the model predicted as positive, how many predictions were correct?

$$
\operatorname{Precision} = \frac{TP}{TP + FP}
$$

Prioritise precision when false positives are particularly costly. For example, a spam filter needs high precision if incorrectly hiding legitimate mail is unacceptable.

## Recall

[Recall](../../../permanent/recall.md) asks: of all the examples that were actually positive, how many did the model find?

$$
\operatorname{Recall} = \frac{TP}{TP + FN}
$$

Prioritise recall when false negatives are particularly costly. For example, an initial disease-screening model should avoid missing people who may have the disease, even if that produces more false alarms for follow-up testing.

## F1 score

The [F1-Score](../../../permanent/f1-score.md) is the harmonic mean of precision and recall:

$$
F_1 = 2 \times \frac{\operatorname{Precision} \times \operatorname{Recall}}{\operatorname{Precision} + \operatorname{Recall}}
$$

The harmonic mean is pulled down strongly when either precision or recall is low. F1 is useful when both kinds of error matter, but it ignores true negatives and assumes precision and recall should contribute equally. It is not a substitute for considering the real cost of each error.

## Multiclass classification

A multiclass confusion matrix has one row and column for every class. Correct predictions appear on the diagonal. Off-diagonal cells show which classes the model confuses with one another.

Precision, recall and F1 can be calculated separately for each class by treating that class as positive and all other classes as negative. Per-class results can then be reported directly or combined using an averaging strategy such as micro, macro or weighted averaging.

## References

- [Confusion matrix, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Precision score, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
- [Recall score, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
- [F1 score, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
