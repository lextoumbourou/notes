---
title: Cross-Entropy Loss Function
date: 2021-07-29 22:00
tags:
  - MachineLearning
  - LossFunctions
---

Cross-entropy is a loss function used in multiclass classification model training. It applies the [[Softmax Activation Function]] to a model's logits before applying the [[Negative Log-Likelihood]] to the value corresponding with the truth label.

Lower Cross-Entropy loss corresponds to results closer to ground truth.

It's expressed as $$-\sum\limits_{i=1}^{N} Y_{i} \times log(P_{i})$$ where $N$ is the total classes, $Y$ is the ground truth labels and $P$ is the softmax probabilities. Since $Y$ is one-hot encoded, you effectively select the prediction corresponding with the ground truth and take its log.

[[Cross-Entropy]] in Information Theory measures the difference between 2 probability distributions.

![Cross-entropy loss function](/_media/cross-entropy-loss-function.png)

[@howardDeepLearningCoders2020] *(pg. 222-231)*