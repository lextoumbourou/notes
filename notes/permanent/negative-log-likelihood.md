---
category: note
title: Negative Log-Likelihood
date: 2021-07-10 00:00
cover: /_media/negative-log-likelihood.png
tags:
- MachineLearning
- LossFunction 
summary: a multi-class classification loss function
notebook:
  markdownLinks: true
---

Negative log-likelihood, or NLL, is a [Loss Function](loss-function.md) used in multi-class classification. It measures how closely our model predictions align with the ground truth labels.

It is calculated as $-log(\hat{y})$, where $\hat{y}$ is the prediction corresponding to the true class label after the model outputs are converted into probabilities by applying the [Softmax Function](softmax-function.md) to them. The loss for a mini-batch is computed by calculating the NLL for each item and then calculating the mean or sum of all items in the batch.

Since a negative value is returned for the log of a number greater than 0 and less than 1, we add a negative sign to convert it to a positive number, hence *negative* log-likelihood. At 0 the function returns $\infty$ ($-log(0)=\infty$) and at 1 returns 0 ($-log(1)=0$), so very wrong answers are heavily penalised.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.01, 1.0, 0.001)
y = -np.log(x)

fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,y)
plt.ylabel('-log(x)')
plt.xlabel('x')
plt.title('Negative log-likelihood range')
plt.show()
```
<!-- nb-output hash="4a628b6d1b745c37" format="html" -->
<div class="nb-output">
<img class="nb-output-image" alt="Line plot of -log(x) for x from 0.01 to 1: the value is about 4.6 near 0, falls steeply, and reaches 0 at x = 1." src="data:image/png;base64,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" />
</div>
<!-- /nb-output -->
Because the [Softmax Function](softmax-function.md) tends to force a single significant number, the loss function only needs to be concerned with the loss corresponding to the correct labels.

In PyTorch, the function is called `torch.functional.nll_loss`, although it doesn't take the log, as it expects outputs from a `LogSoftmax` activation layer.

Referred to as [Log Loss](log-loss.md) in binary classification problems.

Code example:

```python
import numpy as np, pandas as pd
from torch import tensor, nn
```
<!-- nb-output hash="8c92866c6cc2fb29" format="html" -->

<!-- /nb-output -->

```python
def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
```
<!-- nb-output hash="5cb5fe6a178c66cd" format="html" -->

<!-- /nb-output -->

```python
labels = [0, 2, 1, 3]
logits =  np.array([
    [3.5, -3.45, 0.23, 1.25],
    [-2.14, 0.54, 2.67, -5.23],
    [-1.34, 5.01, -1.54, -1.17],
    [ -2.98, -1.37, 1.54,5.23]
])
probs = softmax(logits)
log_probs = np.log(softmax(logits))
```
<!-- nb-output hash="f3fabb2f0e9cbe74" format="html" -->

<!-- /nb-output -->

```python
nll = -(log_probs[range(len(labels)), labels])
nll
```
<!-- nb-output hash="dc30df40cee1ab2f" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">array([0.13484927, 0.11987498, 0.00523357, 0.02625655])
</pre>
</div>
<!-- /nb-output -->

We can check to see that is equal to the output of [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html), with no `mean` or `sum` reduction step.

```python
nn.NLLLoss(reduction='none')(tensor(log_probs), tensor(labels))
```
<!-- nb-output hash="81e10723bc45e99c" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">tensor([0.1348, 0.1199, 0.0052, 0.0263], dtype=torch.float64)
</pre>
</div>
<!-- /nb-output -->

Below the cell is highlighted that corresponds to the correct label.

```python
def style_specific_cell(x):
    return ['background-color:#ccc' if i == x.label else '' for i in range(len(x))]

df = pd.DataFrame(probs, columns=['man', 'woman', 'camera', 'tv'])
df['label'] = labels
df['-log(pred)'] = nll


df.style.apply(style_specific_cell, axis=1)
```
<!-- nb-output hash="6e860feed0eeadcf" format="html" -->
<div class="nb-output">
<div class="nb-output-html"><style>#T_a8653_row0_col0, #T_a8653_row1_col2, #T_a8653_row2_col1, #T_a8653_row3_col3 { background-color: #ccc; }</style>
<table id="T_a8653">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a8653_level0_col0" class="col_heading level0 col0" >man</th>
      <th id="T_a8653_level0_col1" class="col_heading level0 col1" >woman</th>
      <th id="T_a8653_level0_col2" class="col_heading level0 col2" >camera</th>
      <th id="T_a8653_level0_col3" class="col_heading level0 col3" >tv</th>
      <th id="T_a8653_level0_col4" class="col_heading level0 col4" >label</th>
      <th id="T_a8653_level0_col5" class="col_heading level0 col5" >-log(pred)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a8653_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_a8653_row0_col0" class="data row0 col0" >0.873848</td>
      <td id="T_a8653_row0_col1" class="data row0 col1" >0.000838</td>
      <td id="T_a8653_row0_col2" class="data row0 col2" >0.033212</td>
      <td id="T_a8653_row0_col3" class="data row0 col3" >0.092103</td>
      <td id="T_a8653_row0_col4" class="data row0 col4" >0</td>
      <td id="T_a8653_row0_col5" class="data row0 col5" >0.134849</td>
    </tr>
    <tr>
      <th id="T_a8653_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_a8653_row1_col0" class="data row1 col0" >0.007227</td>
      <td id="T_a8653_row1_col1" class="data row1 col1" >0.105412</td>
      <td id="T_a8653_row1_col2" class="data row1 col2" >0.887031</td>
      <td id="T_a8653_row1_col3" class="data row1 col3" >0.000329</td>
      <td id="T_a8653_row1_col4" class="data row1 col4" >2</td>
      <td id="T_a8653_row1_col5" class="data row1 col5" >0.119875</td>
    </tr>
    <tr>
      <th id="T_a8653_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_a8653_row2_col0" class="data row2 col0" >0.001738</td>
      <td id="T_a8653_row2_col1" class="data row2 col1" >0.994780</td>
      <td id="T_a8653_row2_col2" class="data row2 col2" >0.001423</td>
      <td id="T_a8653_row2_col3" class="data row2 col3" >0.002060</td>
      <td id="T_a8653_row2_col4" class="data row2 col4" >1</td>
      <td id="T_a8653_row2_col5" class="data row2 col5" >0.005234</td>
    </tr>
    <tr>
      <th id="T_a8653_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_a8653_row3_col0" class="data row3 col0" >0.000265</td>
      <td id="T_a8653_row3_col1" class="data row3 col1" >0.001325</td>
      <td id="T_a8653_row3_col2" class="data row3 col2" >0.024325</td>
      <td id="T_a8653_row3_col3" class="data row3 col3" >0.974085</td>
      <td id="T_a8653_row3_col4" class="data row3 col4" >3</td>
      <td id="T_a8653_row3_col5" class="data row3 col5" >0.026257</td>
    </tr>
  </tbody>
</table>
</div>
</div>
<!-- /nb-output -->
Negative Log-Likelihood is the 2nd part of the [Categorical Cross-Entropy Loss](categorical-cross-entropy-loss.md).

---

## Recommended Reading

[Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD](https://amzn.to/3Svowuu)

![Deep Learning for Coders with fastai & PyTorch](../_media/deep-learning-for-coders-book-cover.png)

This book is my favourite practical overview of Deep Learning. Learn more about negative log-likelihood in Chapter 6, pg. 231-232.
