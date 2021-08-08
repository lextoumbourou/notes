---
title: Binary Cross-Entropy Loss Function
date: 2021-08-08 17:00
tags:
  - MachineLearning
  - LossFunctions
---

Binary Cross-Entropy (BCE), also known as log loss, is a loss function used in binary or multi-label machine learning training.

Similar to the [[Cross-Entropy Loss Function]] for data with a single class per example. The difference is that it applies the [[Sigmoid Activation Function]] on the model's outputs before the negative log.

For a single (binary) output, the function can be expressed as:

```python
def binary_cross_entropy_single_label(logit, label):
    pred = sigmoid(logit)

    if label == 1:
        return -log(pred)
    
    return -log(1-pred)
```

Or in math:

$$p = \text{Sigmoid}(o)$$
$$L(p, y) = −(𝑦 \times log(𝑝) + (1−𝑦) \times log(1−𝑝))$$

For multi-label outputs, the function takes the mean (or some other reduction method) for each of the log values:

```python
def binary_cross_entropy(logits, labels):
    log_loss_values = [
        binary_cross_entropy_single_label(logit, label)
        for logit, label in zip(logits, labels)]
    return mean(log_loss_values)
```

Which is represented in math as follows:

$$P = \text{Sigmoid}(O)$$
$$L(P, Y) = −\frac{1}{N} \sum\limits_{i=1}^{N} (Y_{i} \times log(P_{i}) + (1− Y_{i}) \times log(1− P_{i}))$$

PyTorch provides the function via the [`nn.BCELoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) class, which expects you first to apply the Sigmoid function to the model's outputs. It's the equivalent of [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) in multi-class classification.

Use [`nn.BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) if your model doesn't perform the Sigmoid Activation Function on the final layer.

{% notebook permanent/notebooks/bce-loss-function.ipynb cells[9:13] %}

[@howardDeepLearningCoders2020] *(pg. 256-257)*