---
title: Negative Log-Likelihood
date: 2021-07-10 15:00
cover: /_media/negative-log-likelihood.png
tags:
  - MachineLearning
  - LossFunctions
---

Negative log-likelihood is a loss function used in multi-class classification.

Calculated as $-log(\textbf{y})$, where $\textbf{y}$ is a prediction corresponding to the true label, after the [[Softmax Function]] was applied. The loss for a mini-batch is computed by taking the mean or sum of all items in the batch.

In Math:

$$\sum\limits_{Y}^{i=1} -\log(Y_i)$$

Since a negative value is returned for the log of a number >= 0 and < 1, we add an additional negative sign to convert to a positive number: hence *negative* log-likelihood. At 0 the function returns $\infty$ ($-log(0)=\infty$) and at 1, ($-log(1)=0$), so very wrong answers are heavily penalised.

{% notebook permanent/notebooks/log-fractions.ipynb %}

Because the [[Softmax Function]] tends to force a single significant number, the loss function only needs to be concerned with the loss corresponding to the correct labels.

In PyTorch, the function is called `torch.functional.nll_loss`, although it doesn't take the log, as it expects outputs from a `LogSoftmax` activation layer.

 [@howardDeepLearningCoders2020] *(pg. 226-231)*

Code example:

{% notebook permanent/notebooks/negative-log-likelihood.ipynb %}

Negative Log-Likelihood is the 2nd part of the [[Cross-Entropy Loss Function]].