---
title: Cross-Entropy Loss Function
date: 2021-07-29 22:00
tags:
  - MachineLearning
  - LossFunctions
---

Cross-entropy is a loss function used in multiclass classification model training. It applies the [[Softmax Activation Function]] to a model's logits before applying the [[Negative Log-Likelihood]] function.

Lower Cross-Entropy loss corresponds to results closer to ground truth.

In math, it's expressed as $$-\sum\limits_{i=1}^{N} Y_{i} \times log(P_{i})$$

$N$ is the number of classes, $Y$ is the ground truth labels, and $P$ is the softmax probabilities. Since $Y$ is one-hot encoded, the labels that don't correspond to the ground truth will be multiplied by 0, so we effectively take the log of only the prediction for the true label.

![Cross-entropy loss function](/_media/cross-entropy-loss-function.png)

Based on [[Cross-Entropy]] in Information Theory, which measures the difference between 2 probability distributions.

[@howardDeepLearningCoders2020] *(pg. 222-231)*