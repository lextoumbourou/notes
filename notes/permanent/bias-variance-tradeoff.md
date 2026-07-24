---
title: Bias-Variance Tradeoff
date: 2024-04-11 00:00
modified: 2026-07-25 07:45
status: draft
video: planned
tags:
- MachineLearning
---

A topic that a Machine Learning practitioner should know, if nothing else for the purposes of passing interviews, is the **bias-variance trade-off**.

The basic idea is that simpler models tend to oversimplify the problem and fail to learn all the signal present in the data (underfitting). More complex models might fit the training data too closely and fail to generalise to new examples (overfitting).


The trade-off implies there is a level of model complexity that minimises expected test error by balancing bias and variance, as [@@fortmannroeUnderstandingBiasVariance2012] shows:

![A U-shaped total error curve produced by decreasing squared bias and increasing variance as model complexity grows.](../_media/bias-variance-tradeoff-fortmann-roe-2012.png)

Note that the error here refers to [Mean-Squared Error](../../../permanent/mean-squared-error.md). Because MSE squares prediction errors, expanding its expected value produces a squared bias term.

---

Main thing to remember:

- **High-Bias, Low-Variance** is associated with **underfitting**
- **High-Variance, Low-Bias** is associated with **overfitting**

![High bias and low variance produce similar but systematically wrong fits, while high variance and low bias produce flexible fits that change substantially with the training data.](../_media/bias-variance-underfitting-overfitting-comparison.png)

---

The word "bias" here comes from the statistical definition: the difference between a model's expected prediction across different training sets and the true value it's trying to estimate.

High bias is often introduced by a model that is too simple to accurately represent the problem.

![Wikipedia defines estimator bias as the difference between an estimator's expected value and the true value of the parameter being estimated.](../_media/bias-of-an-estimator-wikipedia-definition.png)

The word "variance" also comes from the statistical definition of variance: a measure of dispersion. The key idea is that it measures how far a model's predictions would be spread if you trained on different training datasets.

![Wikipedia defines variance as a measure of dispersion and the expected squared deviation from the mean.](../_media/variance-wikipedia-definition.png)

In the worst case, an extremely high-variance model may fit random noise in the training data - and not actually learn the things we care about.

---

This framing was applied to neural networks in the landmark work by [@@gemanNeuralNetworksBias1992], but the idea goes back at least to Grenander's 1952 "uncertainty principle" in statistics [@grenanderEmpiricalSpectralAnalysis1952], with later examples in cubic smoothing splines and a 1990 statistics textbook [@nealBiasVarianceTradeoffTextbooks2019].

In practice, the trade-off is seen as a fallacy - especially in the LLM era. There are many examples of complex models, especially neural networks, where increasing the size of the network can decrease both variance and bias [@nealBiasVarianceTradeoffTextbooks2019].

Test error can also follow a **double-descent** curve: the classical U-shape is followed by a second descent after the model begins to interpolate the training data [@belkinReconcilingModernMachinelearning2019].

Additionally, the kind of LLMs available today are constantly improving their ability to generalise to unseen tasks (or at least - the training sets are so big they encompass nearly everything we can think to test) - so it's not something you hear as much about.

Even so, there are still problems, especially with limited data, where the classical tradeoff is essential to understand.

---

Conventionally, to reduce bias, you might:

- use a more complex model (increase parameters, features, etc)
- reduce regularisation

Conversely, to reduce variance, you might:

- collect more data
- apply additional regularisation
- use a simpler model
- use data augmentation or early stopping
- combine multiple models through ensembling

---

For squared-error regression, the expected prediction error at a given $x$ is often decomposed as:

$MSE = \operatorname{Bias}(\hat{f}(x))^2 + \operatorname{Variance}(\hat{f}(x)) + \sigma^2$

Bias is squared because MSE measures squared error. Bias can be positive or negative, but either direction contributes to prediction error. When the expected squared error is expanded, the systematic difference between the model's expected prediction and the true value therefore appears as:

$\operatorname{Bias}(\hat{f}(x))^2 = \left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2$

This gives squared bias the same units as variance and MSE.

The term $\sigma^2$ represents the irreducible noise in the data that you cannot get rid of with a better model.
