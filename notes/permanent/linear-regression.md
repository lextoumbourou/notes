---
title: Linear Regression
date: 2016-06-28 00:00
modified: 2016-06-28 00:00
status: draft
---

**Linear Regression** is a supervised learning algorithm for determining the relationship between a independent variable (X) and a dependent variable (y). It can also be used to determine the relationship between multiple independent variables and a dependent variable, in which case it's technically called [Multiple Linear Regression](../../../permanent/multiple-linear-regression.md).

The term **Linear** because it uses a [Linear Model](linear-model.md), and [Regression](regression.md) due to the continuous dependent variable.

For example, consider some data that maps the size of a house to its price:

| House size (sq ft) | House price |
| ------------------ | ----------- |
| 250                | $100k       |
| 500                | $150        |

The size of the house is consider the dependant variable, as it's typically plotted on the X-axis. The price of the house is considered the dependant variable, and we plot if on the Y-axis.

{plot goes here}

Linear Regression is about finding the line that best fits the data. This line can be used to make predictions about dependent variables for new datapoints i.e. we can utilise the line to calculate an estimated price given a house size.

{plot with line goes here}

The [Slope-Intercept Form](../../../permanent/Slope-Intercept%20Form.md) of a linear equation is $y = mx + b$, where $y$ = the dependant variable and $x$ is the independent variable. The job of linear regression is to calculate $m$ and $b$. In Linear Regression, b is referred to as the bias term, and $m$ as the weight.

How do we calculate the bias and weight terms? Using the amazing [Gradient Descent](gradient-descent.md) algorithm.

Gradient descent is an iterative algorithm, that starts with a random line, and continuation updates the bias and weights, by making predictions, comparing the predictions to the true labels, calculating the gradients of the errors with respect to the weights, then updating the weights a small amount in the direction of the gradient.

Here's a simple pseudo-code example:

```python
for i in range(num_iterations):
    preds = get_model_predictions()
    error = preds - target
    get_derivative()
    update_weights()
```

Typically we use [Mean-Squared Error](../../../permanent/mean-squared-error.md) as the loss function, since [Mean Absolute Error](mean-absolute-error.md) isn't a differentiable at 0.

In this example of Linear Regression from scratch, for each iteration, I:

get model predictions - which is each feature multiplied by the corresponding cooefficinet. One trick to add the bias therm, is to add a feature that's just the constant 1.

Then, get derivate. The derivative of mean-squared error is,

-2/N * ground_truch - module_prediction * x_iw
