---
title: Week 6 - Precision Recall
date: 2016-07-04 00:00
category: reference/moocs
parent: ml-classification
status: draft
---

## Why use precision & recall as quality metrics

### What is good performance for a classifier?

* Good performance is specific to the task. Sometimes accuracy may not be enough to determine performance.
  * Consider accuracy of 90% in sentiment analyser. If 90% is negative, then it could just be picking 1 class.
* Other performance tools:
  * Precision: did I predict the class correctly?
  * Recall: did I predict all the positive classes as positive?

## Precision & recall explained

### Precision: fraction of positive predictions that are actually positive

* Predicted 10 positive reviews but only 7 of them are positive.
  * Precision: 7/10
  * ``# true positives / (# true positives + # false positives)``
  * Best value = 1.0, worst = 0.0

### Recall: fraction of positive data predicted to be positive

* Recall 10 positive reviews but 15 of them are positive.
  * Recall: 10 / 15
  * ``# true positives / (# true positives + # false negatives)``
  * Best value = 1.0, worst = 0.0

## The precision-recall tradeoff

### Precision-recall extremes

* Optimistic model: high recall, low precision
  * At extreme: predict all as positives all the time. Low precision.
* Pessimistic model: predict positive only when sure.
  * At extreme: say nothing is positive: 100% precision, low recall.

### Precision-recall tradeoff

* Introduces parameter ``t`` which is the probability above which things are considered positive.
  * $$if P(y = +1 | \mathbf{x}_i) > t: \mathbf{y} = +1 $$
  * Optimistic: t set low: $$t = 0.001 $$
  * Pessimistic: t set high: $$t = 0.999 $$

### Precision-recall curve

* Can compare models using "precision at k"
    * Roughly: if you have 5 spots to place a positive article on your website, how many would actually be positive (4/5 = 0.8)
