---
title: NLP Evaluation
date: 2026-03-08 00:00
modified: 2026-09-26 09:10
status: draft
---

There are two key categories of evaluation in Natural Language Processing: **intrinsic** and **extrinsic**.

Intrinsic is the more direct method of evaluating the model. Using a metric like [Perplexity](perplexity.md), we aim to test the capability of the model in isolation. Other examples include:

* Word embeddings: word similarity scores (e.g. comparing to WordSim-353 or SimLex-999 human judgements)
* POS tagger: accuracy on gold-standard tagged corpus.

On the other hand, Extrinsic is task-based, where we put the model into a system, such as a speech recogniser system, and aim to test the capability of the model applied to a task.

* Language model: BLEU score when used in a machine translation system.

## Metrics

[Accuracy](accuracy.md) is a very intuitive and common metric.

Simply for all results, how many were correct predictions? Or formally: (TP + TN) / (TP + FN + FP + TN)

Given the following confusion matrix for a spam classifier:

|                 | Predicted Spam | Predicted Not Spam |
| --------------- | -------------- | ------------------ |
| Actual Spam     | 45             | 15                 |
| Actual Not Spam | 10             | 130                |

Calculate:

- (a) Accuracy
- (b) Precision (for the Spam class)
- (c) Recall (for the Spam class)
- (d) F1-score (for the Spam class)

(45 + 130) / (45 + 130 + 10 + 15)

[Precision](precision.md) refers to a score of how many predicted positives are actually positive.

TP / (TP + FP) = 45 / (45 + 10)

[Recall](recall.md) refers to how many of the actual positives were predicted as positive.

TP / (TP + FN) = 45 / (45 + 15)

[F1-Score](f1-score.md) is the harmonic mean of Precision and Recall.