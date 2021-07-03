---
title: Sigmoid Function
date: 2016-07-01 00:00
cover: /_media/sigmoid-plot.png
tags:
  - MachineLearning
  - MachineLearningMath
---

The Sigmoid function squeezes numbers into a probability-like range between 0 and 1. Used in [[Classification]] model architectures to compute loss on discrete labels, that is, labels that are either 1 or 0 (hotdog or not hotdog).

$$S(x) = \frac{1}{1 + e^{-x}}$$

The intuition is when `x` is infinity ($e^{-\infty}=0$), the Sigmoid becomes $\frac{1}{1}$ and when `x` is -infinity ($e^{\infty} = \infty$) the Sigmoid becomes $\frac{1}{inf}$. That means the model is incentivised to output values as high as possible in a positive case, and low for the negative case. 

[@foxMachineLearningClassification]

It is named Sigmoid because of its S-like [[Function Shape]]. Its name combines the lowercase sigma character and the suffix *[-oid](https://www.dictionary.com/browse/-oid)*, which means *similar to*. 

{% notebook permanent/notebooks/sigmoid-function.ipynb %}

Technically, there are many Sigmoid functions, which output different ranges. For example, the Hyperbolic tangent outputs range between -1 and 1. However, in Machine Learning, the Logistic Function is commonly just referred to as Sigmoid.