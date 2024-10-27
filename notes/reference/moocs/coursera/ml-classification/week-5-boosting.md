---
title: Week 5 - Boosting
date: 2016-07-04 00:00
category: reference/moocs
parent: ml-classification
status: draft
---

## Amazing idea of boosting a classifier

### The boosting question

* Can combine weak/simple classifiers to create strong learners.
* Widely used effective technique.

### Ensemble classifiers

* Rough idea: get a bunch of decision tree stumps that return a sign (+1 or -1) and learn the weights of each stump. Then, add all the weighted stumps to get the final sign.
    * For example, the first feature could be income threshold:
	    * ``+1 if income > 200k else -1``
  	    * Represented as $$f_1(\mathbf{x}_i) = +1 $$
    * Then the second credit history $$f_2(\mathbf{x}_i) = -1 $$

and so on.
    * Then you combine all the classifiers with there weights as follows:

        * $$F(\mathbf{x}_i) = sign(w_1f_1(\mathbf{x}_i) + w_2f_2(\mathbf{x}_i) + w_3f_3(\mathbf{x}_i)) $$

		* Or more succinctly: $$F(\mathbf{x}_i) = sign(\sum\limits_{i=1}^{T} \hat{w}_tf_t(\mathbf{x})) $$

       * Then the weighted features can be used to make predicts.
* Used very frequently in practise.

### Boosting

* Learn data points where you've made mistakes and increase weight of data points.
    * Question: how do you learn the weights of data?
	* Now, with the re weighted data, you add the next classifier. Presumably this means that the next classifier will attempt to correct the mistakes of the first.

## AdaBoost

### AdaBoost Overview

1. Start with same weight for all points: $$\alpha_i = 1/N $$
2. For t = 1, ... T (T = set of classifiers)

	* Learn $$f_t(\mathbf{x}) $$
   * Compute coefficient $$\hat{w}_t $$
   * Recompute weights $$\alpha_i $$
   * Normalise weights $$ \alpha_i $$

3. Final model predicts with: $$F(\mathbf{x}_i) = sign(\sum\limits_{i=1}^{T} \hat{w}_tf_t(\mathbf{x})) $$

* Problems:
    * How do much do you trust the results of the function $$f_t $$

?
    * How do you weigh mistakes more?

### Weighted error

* Basic idea: if one classifier, $$f_t(\mathbf{x}) $$

, is good, you want a large coefficient, if not, small.
	* "Good" = low training error
* Calculate training error as follows:
    * ``sum(total weight of mistakes) / sum(total weight of all data points)``
* Best possible values is 0.0 -> worst 1.0 -> random 0.5

### Computing coefficient of each ensemble component

* Coefficient is calculated with the following formula:

	$$\hat{w}_t = \frac{1}{2} ln( 
		\frac{ 1 - \text{weighted_error}(f_t)}{\text{weighted_error(f_t)}})$$
	




* Interesting thing about the formula is poor classifiers, ones with values close to 1 are inverted to become good classifiers the other way. Average classifiers, eg random, end up with weights of 0.

### Reweighing data to focus on mistakes

$$\alpha_i \leftarrow \begin{cases} \alpha_i\mathrm{e}^{-\hat{w}_t}, \mathrm{if}\space f_t(\mathbf{x}_i) = y_i \\ \alpha_i\mathrm{e}^{\hat{w}_t}, \mathrm{if}\space f_t(\mathbf{x}_i) \neq y_i \end{cases} $$

* Results:
	* When $$f_t(\mathbf{x}_i) $$

is correct (and thus we have a high weight for it) we end up with a lower weight for the datapoint.
   * When the opposite, we have high weights for data point, this means that the data points that are wrong will be more important to future classifiers.
   * When results are average, we keep the data point the same (weight of 1).

### Normalising weights

* If $$\mathbf{x}_i $$

often mistakes, weight $$\alpha_i $$

gets large and vice versa.
* Can cause numerical instability.
* Solution: normalise each data point by ensuring all weights add up to 1 after each iteration:
	* $$\alpha_i \leftarrow \frac{\alpha_i}{\sum\limits_{i=j}^{N} \alpha_j}  $$

## Convergence and overfitting in boosting

* Boosting can be less sensitive to overfitting.
* Decide when to stop boosting using validation set or cross-validation (never use test data or, obviously, training data)

# Summary

## Ensemble methods, impact of boosting and quick recap

* Gradient boosting: similar to AdaBoost but can be used beyond classification
* Random forests: "Bagging" picks random subsets of data and learns tree in each subset.
	* Good for parallelising.
    * Simple.
