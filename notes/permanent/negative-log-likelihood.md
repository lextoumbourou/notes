---
title: Negative Log-Likelihood
aliases: Log Loss
date: 2021-07-10 00:00
cover: /_media/negative-log-likelihood.png
tags:
  - MachineLearning
---

Negative log-likelihood is a loss function used in multi-class classification.

It is calculated as $-log(\textbf{y})$, where $\textbf{y}$ is the prediction corresponding to the true label after the [Softmax Activation Function](softmax-activation-function.md) was applied. The loss for a mini-batch is computed by taking the mean or sum of all items in the batch.

Since a negative value is returned for the log of a number >= 0 and < 1, we add a negative sign to convert it to a positive number, hence *negative* log-likelihood. At 0 the function returns $\infty$ ($-log(0)=\infty$) and at 1 returns 0 ($-log(1)=0$), so very wrong answers are heavily penalised.

{% notebook permanent/notebooks/log-fractions.ipynb %}

Because the [Softmax Activation Function](softmax-activation-function.md) tends to force a single significant number, the loss function only needs to be concerned with the loss corresponding to the correct labels.

In PyTorch, the function is called `torch.functional.nll_loss`, although it doesn't take the log, as it expects outputs from a `LogSoftmax` activation layer.


Code example:

{% notebook permanent/notebooks/negative-log-likelihood.ipynb %}

Negative Log-Likelihood is the 2nd part of the [Categorical Cross-Entropy Loss](Categorical Cross-Entropy Loss.md).

---

## Recommended Reading

[Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD](https://amzn.to/3Svowuu)

![Deep Learning for Coders with fastai & PyTorch](../_media/deep-learning-for-coders-book-cover.png)

This book is my favourite practical overview of Deep Learning. Learn more about negative log-likelihood in Chapter 6, pg. 231-232.