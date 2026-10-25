---
category: note
title: Information Entropy
date: 2021-07-28 00:00
tags:
  - InformationTheory
cover: /_media/claude-shannon-entropy.jpeg
summary: A measure of uncertainty — highest when all outcomes are equally likely, lowest when one dominates.
notebook:
  markdownLinks: true
---

Entropy is a measure of uncertainty of a random variable's possible outcomes.

It's highest when there are many equally likely outcomes. As you introduce more predictability (one of the possible values of a variable has a higher probability), Entropy decreases.

It measures how many "questions" on average you need to guess a value from the distribution. Since you'd start by asking the question that is most likely to get the correct answer, distributions with low Entropy would require smaller message sizes on average to send.

The entropy of a variable from distribution $p$ is expressed as: $$H=\sum\limits_{i=1}^{n} p_{i} \times log_2(\frac{1}{p})$$

 The expression is commonly inverted and rewritten like this: $$H=-\sum\limits_{i=1}^{n} p_{i} \times log_2(p_i)$$

When using log base 2, the unit of Entropy is a bit (a yes or no question).

In code:

```python
import numpy as np
```
<!-- nb-output hash="d37e32faa7933d9c" format="html" -->

<!-- /nb-output -->

```python
def entropy(dist):
    return -np.sum(np.array(dist) * np.log2(dist))
```
<!-- nb-output hash="5d436c2e0b58db30" format="html" -->

<!-- /nb-output -->

When the variable has a 50/50 distribution, you will always need to ask one question to find the answer:

```python
dist = [1/2.]*2
dist
```
<!-- nb-output hash="67b2ada318a70986" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">[0.5, 0.5]
</pre>
</div>
<!-- /nb-output -->

```python
entropy(dist)
```
<!-- nb-output hash="8fa00ed5838e24a4" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">np.float64(1.0)
</pre>
</div>
<!-- /nb-output -->

When you have only one possible outcome, you don't need to ask any questions:

```python
dist = [1.]
dist
```
<!-- nb-output hash="09dd62d5906e0a09" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">[1.0]
</pre>
</div>
<!-- /nb-output -->

```python
entropy(dist)
```
<!-- nb-output hash="8fa00ed5838e24a4" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">np.float64(-0.0)
</pre>
</div>
<!-- /nb-output -->

When you have a lot of equally likely possibilities, you have to ask a lot of questions and entropy is high:

```python
dist = [1/100.]*100
dist[:5]
```
<!-- nb-output hash="1e74e1d7a85e8569" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">[0.01, 0.01, 0.01, 0.01, 0.01]
</pre>
</div>
<!-- /nb-output -->

```python
entropy(dist)
```
<!-- nb-output hash="8fa00ed5838e24a4" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">np.float64(6.643856189774724)
</pre>
</div>
<!-- /nb-output -->

When you have one really likely possiblity, you only need to ask a question when the answer is the unlikely case:

```python
dist = [99/100] + [1/100.]
dist
```
<!-- nb-output hash="f427d6cdb3ce53f9" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">[0.99, 0.01]
</pre>
</div>
<!-- /nb-output -->

```python
entropy(dist)
```
<!-- nb-output hash="8fa00ed5838e24a4" format="html" -->
<div class="nb-output">
<pre class="nb-stream-stdout">np.float64(0.08079313589591118)
</pre>
</div>
<!-- /nb-output -->
Claude Shannon borrowed the term [Entropy](entropy.md) from thermodynamics as part of his theory of communication.

[@khanacademylabsInformationEntropyJourney]

Cover from [How Claude Shannon Invented the Future](https://www.quantamagazine.org/how-claude-shannons-information-theory-invented-the-future-20201222/).
