---
title: Root mean-squared error - L2 Loss
date: 2016-02-03 00:00
tags:
  - MachineLearning
---

Root mean-squared error (RMSE) is a function for assessing [Regression](Regression) predictions. Sometimes called L2 Error because it takes the [L2 Norm](L2 Norm) of the error vector.

### Steps

1. Calculate error vector as **labels** - **predictions**
2. Square the errors to make values positive.
3. Take the mean of all values.
4. Take the square root of the mean.

$$\sqrt{\frac{1}{n} \sum\limits_{i=1}^{N} |{labels}-{predictions}|}$$

An alternative to [Mean Absolute Difference - L1 Loss](mean-absolute-error-L1-loss.md), where the key difference is the square operation instead of the absolute value. Because the square operations penalises very wrong predictions, it is often preferred. However, some also say it can be sensitive to outliers. In my experience, it's usually worth trying both (or some combination).

Sometimes you choose not to perform the square root operation when in a loss function. In that case, the operation is called a **squared** [L2 Norm](L2 Norm) and expressed $\| e\|^{2}_{2}$

{% notebook permanent/notebooks/rmse-example.ipynb %}

#### References

* [What Are L1 and L2 Loss Functions](https://afteracademy.com/blog/what-are-l1-and-l2-loss-functions#:~:text=L2 Loss Function is used,value and the predicted value.)
* [L2 loss vs. mean squared loss, 2nd answer](https://datascience.stackexchange.com/questions/26180/l2-loss-vs-mean-squared-loss)
