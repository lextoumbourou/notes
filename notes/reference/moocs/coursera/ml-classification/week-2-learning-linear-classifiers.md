---
title: Week 2 - Learning Linear Classifiers
date: 2016-07-04 00:00
category: reference/moocs
parent: ml-classification
status: draft
---

## Data likelihood

* Quality metric for logistic regression
* Want probability to be close to correct classification for some $\hat{w}$ and features.
* Data likelihood = Calculate probability that result equals the actual output for each data point in the training dataset then take product of results:

	$$l(w) = P(y_1 | \mathbf{x}_1,w) * P(y_2 | \mathbf{x}_2,w) ... * P(y_N | \mathbf{x}_N,w)$$

	$$l(w) = \prod\limits_{i=1}^{N} P(y_i | \mathbf{x}_i, \mathbf{w})$$

	* Note: you multiply the probabilities because you assume probabilities of each data point are independent.

## Review of gradient ascent

* Algorithm for one-dimensional space:
	* while not converged: $w^{(t+1)} < w^{(t)} + \eta \frac{dl}{d}$
* Rough intuition: step towards a maximum multiplying the derivative by a step size, until the derivate is basically 0, which means you have hit the maximum. If you over step, then you should be pulled in the opposite direction, meaning you should eventually converge.
* When multi-dimension, replace derivative with a D-dimensional vector of partial derivatives:

    $$\triangledown l(\mathbf{w}) =\begin{bmatrix}\frac{\partial l}{\partial w_0}\\\frac{\partial l}{\partial w_1}\\\frac{\partial l}{\partial w_D}\end{bmatrix}$$

## Learning algorithm for logistic regression

Derivative of (log-)likelihood:

    $$\frac{\partial l(\mathbf{w})}{\partial \mathbf{w}_j} = \sum\limits_{i=1}^{N} h_j(\mathbf{x}_i)(\mathbf{1}[y_i = +1] - P(y = +1 | \mathbf{x}_i, \mathbf{w})]]) $$

Rough attempt at writing gradient ascent:

    #...
    
    converged = False
    while not converged:
       for j in range(len(weights)):
           partial[j] = (d[j] * (get_indicator(data) - get_probability(data, coefficients)).sum()
       coefficients = step_size * partial
       converged = assess_convergence_somehow(partial)  # ??

## Interpreting derivative for logistic regression

* Contribution for a single datapoint should work as follows:
	* When probability output is close to 1, the partial for that coefficient should be close to 0:
		* For example, if the probability is 1, the partial would be calculated like: (1 - 1) = 0
			* If the probability was way off, it would be closer to 1 and push the coefficient value up.
		* On the other hand, if it's a negative datapoint and we get it wrong: (0 - 1) = -1 which should push the coefficient in the negative direction.
			* If we get it right, the probability of it being positive should be near 0, so: (0 - 0), which would keep the coefficient the same.

## Choosing the step size η

* Too small step size: too many iterations until convergence.
* Too high step size: overstep the convergence point.

## Rule of thumb for choosing step size

* Requires lots of trial and error: hard.
* Try several values, exponentially spaced:
	* Find one that's too small.
	* Find one that's too large.
	* Try values inbetween.

## Evaluating a classifier

* Standard: Split training data into training set and validation set.
* Figure out how many mistakes the model makes on the validation set.
* Then classification error:

      error = num mistakes / total number of datapoints

  * Best possible value is 0: `0 / # number of datapoints.`
* Or classification accuracy:

      accuracy = num correct / total number of datapoints

## Review of overfitting in regression

* Overfitting if there exists w*:

      training_error(w*) > training_error(ŵ) and true_error(w*) < true_error(ŵ)

* In other words, you've overfitted if you've got a model that works crazy well on your training data set but shit on validation set (representing true error), but there's another model that does less well on training data and better on validation set.
* Can't expect to get 100% accuracy with your model: getting "everything right" should be a warning sign.
* Large coefficients values are a warning sign.

## Overfitting in classification

* Use classification error (`# num mistakes / # num datapoints`) instead of RSS.
* Same as regression, aim to weigh lower coefficients higher.
* Can lead to extremely over confident predictions: high score, high sigmoid (close to 1) == 100% confidence in bullshite.

## Penalising large coefficients to mitigate overfitting

* Idea similar to l2/l1 penalisation in regression course: need a way to penalise large coefficients in Quality Metric.
* Firstly, measure of fit = log of the data likelihood function:

    $$l(\mathbf{w}) = ln \prod_\limits{i=1}^{N} P(y_i | \mathbf{x}_i, \mathbf{w}) $$

* L2 norm: sum of squares of coefficients:

   $$||w||_2^2 = w_0^2 + w_1^2 + w_2^2 + ... w_D^2 $$

* L1 norm: sum of absolute value of coefficients:

    $$||w||_1 = |w_0| + |w_1| + |w_2| + ... |w_D| $$

* Final quality metric:

        total_quality = measure_of_fit - measure_of_coefficient_magnitude

## L2 regularised logistic regression

* Want to choose a tuning parameter or $\lambda$ value to balance fit and magnitude of coefficients
* Consider extreme choices:
	* $\lambda = 0$: data likelihood function wins, no coefficient magnitude parameter
		* Low bias, high variance.
	* $\lambda = \infty$: coefficients are shrunk to 0.
		* High bias, low variance.
* Find best lambda with "L2 regularised logistic regression"
	* Use validation set if you have enough data.
	* Cross-validation for smaller datasets.

## Visualising effect of L2 regularisation in logistic regression

* High lambda values, eg a higher penalty for large coefficients in your model, can clean up a decision boundary, even when you use heaps of features.
* Can help with overfitting a lot.
* Questions: when do I perform the L2 regularisation on my model?

## Learning L2 regularised logistic regression with gradient ascent?

* Standard gradient ascent but include the derivate of the L2 value in the equation.
    * while not converged: $\mathbf{w^{(t+1)}} \leftarrow \mathbf{w^{(t)}} + \eta \triangledown l(\mathbf{w^{(t)}})$

## Sparse logistic regression with L1 regularisation

* Useful for efficient: forces some coefficients to 0, allowing your predictions to ignore the majority of coefficients (useful when you have large feature sets).
* Total quality:
* measure of fit - l1 penalty of coefficients
