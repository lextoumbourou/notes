---
title: Perplexity
date: 2024-03-12 00:00
modified: 2026-09-26 08:55
status: draft
---

Perplexity is a measure of how well a probability model predicts a sample, often used in natural language processing to evaluate language models.

Lower perplexity indicates better performance.

$PP(W) = P(W)^{-\frac{1}{n}}$

where $n$ is the number of words in the sequence $W$.

For example, for the sentence "the cat sat", if the model gives the words probabilities of 0.5, 0.4 and 0.3:

$P(W) = 0.5 \times 0.4 \times 0.3 = 0.06$

Perplexity formula:

$PP(W) = 0.06^{-\frac{1}{3}} = (1/0.06)^{\frac{1}{3}}$
$PP(W) = 16.67^{\frac{1}{3}}$
$PP(W) \approx 2.55$

Interpretation: model is "confused" about choosing between 2.55 equally likely words at each step.

Lower perplexity: better language model.

* 1 - perfect prediction
* 2-10 - very good
* 50-100 - weak model.
