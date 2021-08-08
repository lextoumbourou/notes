---
title: Binary Cross-Entropy Loss Function
date: 2021-08-08 17:00
tags:
  - MachineLearning
  - LossFunctions
---

Binary Cross-Entropy (BCE), also known as log loss, is a loss function used in binary or multi-label machine learning training.

The critical difference between BCE and the [[Cross-Entropy Loss Function]] is that we apply [[Sigmoid Activation Function]] to each of the model's outputs before $-log$.

For a single binary output, the function can be expressed as:

{% notebook permanent/notebooks/bce-loss-function.ipynb cells[0:2] %}

Or in math:

$$p = \text{Sigmoid}(o)$$
$$L(p, y) = −(\underbrace{y \times log(𝑝)}_{\text{Exp 1}} + \underbrace{(1−𝑦) \times log(1−𝑝)}_{\text{Exp 2}})$$ 

Where $o$ is the model output, $y$ is the true label. Since $y$ will either be `1` or `0`, $\text{Exp 1}$ or $\text{Exp 2}$ will be 0, ensuring we only keep one $\log$ value. That's equivalent to the `if` statement in code.

For multi-label outputs, the function takes the mean (or some other reduction method) for each of the log values:

{% notebook permanent/notebooks/bce-loss-function.ipynb cells[2:4] %}

That is represented in math as follows:

$$P = \text{Sigmoid}(O)$$
$$L(P, Y) = −\frac{1}{N} \sum\limits_{i=1}^{N} (Y_{i} \times log(P_{i}) + (1− Y_{i}) \times log(1− P_{i}))$$

PyTorch provides the function via the [`nn.BCELoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) class, which expects you first to apply the Sigmoid function to the model's outputs. It's the equivalent of [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) in multi-class classification.

Use [`nn.BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) if your model doesn't perform the [[Sigmoid Activation Function]] on the final layer.

{% notebook permanent/notebooks/bce-loss-function.ipynb cells[4:5] %}

which is equivalent to this function:

{% notebook permanent/notebooks/bce-loss-function.ipynb cells[5:7] %}

[@howardDeepLearningCoders2020] *(pg. 256-257)*