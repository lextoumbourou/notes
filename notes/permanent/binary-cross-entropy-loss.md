---
category: note
title: Binary Cross-Entropy Loss
date: 2021-08-08 00:00
tags:
  - MachineLearning
summary: A loss function for binary and multi-label classification, also known as log loss.
notebook:
  markdownLinks: true
---

Binary Cross-Entropy (BCE), also known as log loss, is a loss function used in binary or multi-label machine learning training.

It's nearly identical to [Negative Log-Likelihood](negative-log-likelihood.md) except it supports any number of positive labels (including zero).

For each value in a set of model outputs, we first apply the [Sigmoid Function](sigmoid-function.md) before taking `-log(pred)` if the corresponding label is positive or `-log(1-pred)` if negative.

For a single binary output, the function can be expressed as:

```python
from math import log

def binary_cross_entropy_single_label(pred, label):
    if label == 1:
        return -log(pred)

    return -log(1-pred)
```
<!-- nb-output hash="fd9d7baea67b4d95" format="html" -->

<!-- /nb-output -->

```python
pred = 0.99
label = 1

binary_cross_entropy_single_label(pred, label)
```
<!-- nb-output hash="2946de92a2ce831b" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">0.01005033585350145
</pre>
</div>
<!-- /nb-output -->
Or in math:

$$L(p, y) = -(\underbrace{y \times log(𝑝)}_{\text{Expr 1}} + \underbrace{(1-𝑦) \times log(1-𝑝)}_{\text{Expr 2}})$$

Where $p$ is the model's predictions and $y$ is the true label.

Since $y$ will either be $1$ or $0$, $\text{Expr 1}$ or $\text{Expr 2}$ will be 0, ensuring we only keep one $\log$ value. That's equivalent to the `if` statement in code.

For multi-label outputs, the function takes the mean (or sometimes sum) of each of the log values:

```python
from numpy import mean

def binary_cross_entropy(preds, labels):
    return mean([
        binary_cross_entropy_single_label(pred, label) for pred, label in zip(preds, labels)])
```
<!-- nb-output hash="af7d22c9eaeb1682" format="html" -->

<!-- /nb-output -->

```python
preds = [0.99, 0.05, 0.95]
labels = [1, 0, 1]

binary_cross_entropy(preds, labels)
```
<!-- nb-output hash="a9f688d4bf833785" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">np.float64(0.03754564154286754)
</pre>
</div>
<!-- /nb-output -->
That is represented in math as follows:

$$L(P, Y) = -\frac{1}{N} \sum\limits_{i=1}^{N} (Y_{i} \times log(P_{i}) + (1- Y_{i}) \times log(1- P_{i}))$$

PyTorch provides the function via the [`nn.BCELoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) class. It's the equivalent of [`nn.NLLLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) in multi-class classification with a single true label per input.

```python
from torch import tensor, where, nn

preds = tensor([0.99, 0.05, 0.95]).float()
labels = tensor([1, 0, 1]).float()

nn.BCELoss()(preds, labels)
```
<!-- nb-output hash="fcd565ec48aecb08" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">tensor(0.0375)
</pre>
</div>
<!-- /nb-output -->
which is equivalent to this function:

```python
def binary_cross_entropy(preds, labels):
    return -where(labels==1, preds, 1-preds).log().mean()
```
<!-- nb-output hash="1dd73181aa793512" format="html" -->

<!-- /nb-output -->

```python
binary_cross_entropy(preds, labels)
```
<!-- nb-output hash="5edf51565ef6f25a" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">tensor(0.0375)
</pre>
</div>
<!-- /nb-output -->
Use [`nn.BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) if your model architecture doesn't perform the [Sigmoid Function](sigmoid-function.md) on the final layer. That's equivalent to [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) in PyTorch (see [Categorical Cross-Entropy Loss](categorical-cross-entropy-loss.md)).

[@howardDeepLearningCoders2020] *(pg. 256-257)*
