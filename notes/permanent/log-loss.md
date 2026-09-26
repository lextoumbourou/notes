---
title: Log Loss
date: 2024-04-02 00:00
modified: 2026-09-26 08:58
status: draft
---

Log Loss, or **logistic loss** or **cross-entropy loss** - is a specific case of [Negative Log-Likelihood](negative-log-likelihood.md) for binary classification problems.

The formula for a single data point is: $-(y \times \log(\hat{y}) + (1 - y) \times \log(1 - \hat{y}))$ which is equivalent to:

$$
\begin{cases}
-\log(\hat{y}) & \text{if } y = 1\\
-\log(1 - \hat{y}) & \text{if } y = 0
\end{cases}
$$

To calculate the log loss for an entire dataset, you take the average of each datapoint: $LogLoss = -\frac{1}{n} \sum (y \times \log(\hat{y}) + (1 - y) \times \log(1 - \hat{y}))$

Log Loss is the same as negative log-likelihood after converting binary into multi-class by one-hot encoding the binary labels. 

Since the log of a value between 0 and 1 is negative, we add the negative sign to convert it into a positive number.