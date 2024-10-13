---
title: Mean Absolute Error
date: 2016-02-03 00:00
tags:
  - MachineLearning
---

Mean Absolute Error (MAE) is a metric for assessing [Regression](regression.md) predictions. Simply take the average of the absolute error between all labels and predictions in the test set:

$\frac{1}{N}\sum\limits_{j=1}^{n} |y_i - \hat{y}_i|$

Step-by-step:

1. Calculate error vector as **labels** - **predictions**
2. Take the absolute values of the errors
3. Take the mean of all values.

 It's also known as [L1 Loss](l1-loss.md) or [L1 Penalty](../../../permanent/l1-penalty.md) because it takes the [l1-norm](../../../permanent/l1-norm.md) of the error vector

An alternative to [Root Mean-Squared Error](root-mean-squared-error.md).

#### References

* [What Are L1 and L2 Loss Functions](https://afteracademy.com/blog/what-are-l1-and-l2-loss-functions#:~:text=L2 Loss Function is used,value and the predicted value.)
