---
title: Structuring Machine Learning Projects (Coursera) - Week 1
date: 2017-09-30 00:00
link: 
category: reference/moocs
status: draft
parent: machine-learning-projects
---

## Introduction to ML Strategy

## Size of the dev and test sets

* In traditional ML, you had train / test splits of 70%/30% or 60/20/20. Useful when you have training sets from 100 to 10k.
* Modern ML era: if you have 1M examples, may split 98/1/1, since 1% of 1M is 10k examples.
* Purpose of test set is to evaluate how good your system is: 10k should be enough for that.
  * Some applications may not require a test dev. Just train/dev.

## When to change dev/test sets and metrics

* May need to introduce a penalty for certain outcomes if it fits a business need.
* Example: 5% accuracy vs 3% accuracy with a > of seeing porn.
  * Develop an evaluation metric that incorporates a penalty for seeing porn.
* Have 2 distinct steps when building model:
    1. Place target - decide on a error metrics.
    2. Shot at target - choose algorithms, regularisation, normalisation etc.

* Pick an evaluation metric and change it later if need be.

## Comparing to human-level performance

### Why human-level performance?

* Bayes optimal error: some theoretical maximum performance, potentially above human-level error.
* Optimal error may not be 100%: blurry images, noisy audio etc.
* Humans are good at a lot of tasks, as long as ML is worse than humans, you can:
    1. Get labeled data from humans.
    2. Gain insight from error analysis: what did a person get right?
    3. Better analysis of bias/variance.

### Avoidable bias

* Want to know human-level accuracy to determine if algorithm is doing "too well" on the training set.
* If training accuracy is well below human-level, have a bias problem.
* Consider human-level error a proxy or estimation for Bayes error.
* Difference between Bayes error and training error considered "avoidable bias".
	* Don't want to do better than Bayes error, else you're overfitting.

### Understanding human-level performance

* Medical image classification scenario:

  1. Typical person gets 3% error.
  2. Typical doctor gets 1% error.
  3. Experienced doctor gets 0.7% error.
  4. Team of doctors gets 0.5% error.

  * How would you define "human-level" error?
      * Bayes error estimate would be considered 0.5%.
      * Important to know when your model can do better.

### Surpassing human-level performance

* When your algorithm surpasses human-level accuracy, it becomes harder to figure out how to improve algorithm: options become less clear.
* Problems where ML has surpassed human-level accuracy:
    * Online advertising (predicting if a user will click).
    * Product recommendations.
    * Logistics (predicting transit time).
    * Loan approvals.
* All examples require a big database of information.
	* Not natural perception tasks which humans tend to excel at.

### Improving your model performance

* Two fundamental assumptions:
    1. You can fit the training set well - low avoidable bias.
    2. Training set performance generalises well to dev/test test - low variance.

* Start by looking at avoidable bias - how much better should you be aiming to do on training set?
    * Train a bigger model.
    * Train longer and use better optimization algorithm.
		* Momentum, RMSProp, Adam etc
    * Find better NN architecture / hyperparams.
        * More layers and more hidden units.
        * Try other model architectures.
* Then, start looking at dev error / variance.
	* More data.
	* Regularization.
		* L2, Dropout, Data aug.
	* Better NN architecture / hyperparams.
