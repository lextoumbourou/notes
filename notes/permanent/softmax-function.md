---
title: Softmax Function
date: 2021-07-08 00:00
cover: /_media/softmax-table.png
tags:
  - MachineLearning
  - ActivationFunction
summary: Converts a vector of numbers into probabilities that sum to 1.
---

The Softmax [Activation Function](activation-function.md) converts a vector of numbers into a vector of probabilities that sum to 1. It's applied to a model's outputs (or [Logits](Logits)) in [Multi-class Classification](Multi-class Classification).

It is the multi-class extension of the [Sigmoid Activation Function](Sigmoid Activation Function.md).

 The equation is:

 $$\sigma(\vec{z})_{i} = \frac{e^{z_i}}{\sum\limits_{j=1}^{K}e^{z_j}}$$

 The intuition for it is that $e^{x_i}$ is always positive and increases fast, amplifying more significant numbers. Therefore, it tends to find a single result and is less useful for problems where you are unsure if inputs will always contain a label. For that, use multiple binary columns with the [Sigmoid Activation Function](Sigmoid Activation Function.md).

 [@howardDeepLearningCoders2020] *(pg. 223-227)*

 Code example:

```python
import numpy as np
import pandas as pd

def softmax(x):
    return np.exp(x) / np.exp(x).sum()
    
logits =  np.array([-3.5, -2.37, 1.54, 5.23]) # some arbitrary numbers I made up that could have come out of a neural network
probs = softmax(logits)

df = pd.DataFrame({'logit': logits, 'prob': probs}, index=['woman', 'man', 'camera', 'tv'])
df
```
<!-- nb-output hash="c7a4398df7a8bc35" format="html" -->
<div class="nb-output">
<div class="nb-output-html"><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logit</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>woman</th>
      <td>-3.50</td>
      <td>0.000158</td>
    </tr>
    <tr>
      <th>man</th>
      <td>-2.37</td>
      <td>0.000488</td>
    </tr>
    <tr>
      <th>camera</th>
      <td>1.54</td>
      <td>0.024348</td>
    </tr>
    <tr>
      <th>tv</th>
      <td>5.23</td>
      <td>0.975007</td>
    </tr>
  </tbody>
</table>
</div></div>
</div>
<!-- /nb-output -->

Softmax is part of the [Categorical Cross-Entropy Loss](categorical-cross-entropy-loss.md), applied before passing results to [Negative Log-Likelihood](permanent/Negative Log-Likelihood.md) function.

[Temperature Scaling](temperature-scaling.md) can be applied to the logits to adjust the distribution sharpness (how confident it is about high values) or make it flatter and more diverse.
