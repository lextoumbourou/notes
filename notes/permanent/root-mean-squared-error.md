---
category: note
title: Root Mean-Squared Error
date: 2016-02-03 00:00
tags:
  - MachineLearning
summary: A regression metric that penalises large errors.
notebook:
  markdownLinks: true
---

Root mean-squared error (RMSE) is a function for assessing [Regression](regression.md) predictions.

Sometimes called [L2 Loss](l2-loss.md) because it takes the [L2 Norm](l2-norm.md) of the error vector.

### Steps

1. Calculate error vector as **labels** - **predictions**
2. Square the errors to make values positive.
3. Take the mean of all values.
4. Take the square root of the mean.

$\sqrt{\frac{1}{n} \sum\limits_{i=1}^{N} (\text{labels}-\text{predictions})^2}$

An alternative to [Mean Absolute Error](mean-absolute-error.md), where the key difference is the square operation instead of the absolute value. Because the square operations penalises very wrong predictions, it is often preferred. However, some also say it can be sensitive to outliers. In my experience, it's usually worth trying both (or some combination).

Sometimes you choose not to perform the square root operation when in a loss function. In that case, the operation is called a **squared** [L2 Norm](l2-norm.md) and expressed $\| e\|^{2}_{2}$

In this Numpy example, the model predicted the price of 3 houses (in $) as 350k, 450k and 500k.

The houses are actually valued at 300k, 480k and 800k.

```python
import numpy as np

predictions = np.array([350, 450, 500])
labels = np.array([300, 480, 800])

difference = predictions - labels
difference_squared = difference ** 2

rmse = np.sqrt(difference_squared.mean())
rmse
```
<!-- nb-output hash="f7a7c96f5a484958" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">np.float64(176.44640357154728)
</pre>
</div>
<!-- /nb-output -->
#### References

* [What Are L1 and L2 Loss Functions](https://afteracademy.com/blog/what-are-l1-and-l2-loss-functions#:~:text=L2 Loss Function is used,value and the predicted value.)
* [L2 loss vs. mean squared loss, 2nd answer](https://datascience.stackexchange.com/questions/26180/l2-loss-vs-mean-squared-loss)
