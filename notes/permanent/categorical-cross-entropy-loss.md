---
title: Categorical Cross-Entropy Loss
aliases: [Softmax Loss]
date: 2021-07-29 00:00
tags:
  - MachineLearning
---

Categorical Cross-Entropy Loss Function, also known as Softmax Loss, is a loss function used in multiclass classification model training. It applies the [Softmax Activation Function](softmax-activation-function.md) to a model's output (logits) before applying the [Negative Log-Likelihood](negative-log-likelihood.md) function.

Lower loss means closer to the ground truth.

In math, expressed as

$$P = \text{softmax}(O)$$

$$-\sum\limits_{i=1}^{N} Y_{i} \times \log(P_{i})$$

where $N$ is the number of classes, $Y$ is the ground truth labels, and $O$ is the model outputs. Since $Y$ is one-hot encoded, the labels that don't correspond to the ground truth will be multiplied by 0, so we effectively take the log of only the prediction for the true label.

![Cross-entropy loss function](/_media/cross-entropy-loss-function.png)

{% notebook permanent/notebooks/cross-entropy-pytorch.ipynb %}

Based on [Cross-Entropy](cross-entropy.md) in Information Theory, which is a measure of difference between 2 probability distributions.

[@howardDeepLearningCoders2020] *(pg. 222-231)*
