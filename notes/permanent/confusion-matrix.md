---
title: Confusion Matrix
date: 2024-04-08 00:00
modified: 2024-04-08 00:00
status: draft
---

**Confusion Matrix** is a table that represents the actual and predicted classes, and is a summary of the performance of a classification model on test data.

|                            | Actual Positive | Actual Negative |
| -------------------------- | --------------- | --------------- |
| **Predicted Positive**     | TP              | FP              |
| **Predicted Negative**<br> | FN              | TN              |

**False Positive** is called Type 1 error.

**False Negative** is called Type 2.

This meme has helped me remember Type I and Type II error more than any other technique.

![](/_media/confusion-matrix-type-i-and-type-ii.png)

For example, if we had a table predicting default or not, as follows:

| Actual | Pred |
| ------ | ---- |
| 1      | 0    |
| 1      | 0    |
| 0      | 0    |
| 1      | 1    |

|                    | Actual Positive | Actual Negative |
| ------------------ | --------------- | --------------- |
| Predicted Positive | 3               | 0               |
| Predicted Negative | 2               | 1               |

From that we can calculate a series of metrics:

* Recall:
    * True Positives / True Positives + False Negative
        * Out of all the actual positive cases, how many did we actual find?
        * Out of all the defaults, how many did we find?
        * Out of all the spam emails, how many did we find?
* Precision
    * True Positives / True Positives + False Positives
        * Out of all the positive case predicted, how many did we get right?
* F1 Score:
    * Harmonic mean of recall and precision.

It can be abstracted to multiple classes, if we think of true and false as a confusion matrix with 2 classes, we can simply have more classes.

|              | Is Frog | Is Dog | Is Cat |
| ------------ | ------- | ------ | ------ |
| Predict Frog | 5       | 1      | 0      |
| Predict Dog  | 1       | 5      | 1      |
| Predict Cat  | 2       | 1      | 5      |
