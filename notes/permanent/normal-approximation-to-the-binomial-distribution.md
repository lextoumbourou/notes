---
title: Normal Approximation to the Binomial Distribution
date: 2026-10-10 20:49
modified: 2026-10-10 20:50
status: draft
summary: When and how to estimate binomial probabilities with a normal curve.
tags:
  - Statistics
  - Probability
---

When a [Binomial Distribution](binomial-distribution.md) has enough expected successes and failures, its probability bars form a shape close to a [Normal Distribution](normal-distribution.md). We can then estimate a binomial probability using the area under a normal curve. This is an **approximation**, not a change to the underlying binomial model [@pennStateNormalApproximationBinomial].

For $X\sim\operatorname{Binomial}(n,p)$, the mean is $\mu=np$ and the standard deviation is $\sigma=\sqrt{np(1-p)}$. A conservative rule of thumb is to use the approximation when both $np\geq10$ and $n(1-p)\geq10$. We **standardise** a cutoff $x$ by computing:

$$
z=\frac{x-np}{\sqrt{np(1-p)}}.
$$

This $z$ says how many standard deviations the cutoff lies from the mean. It is a location on the standard normal curve, **not a probability**. The probability is the area to the left or right of that location, found with a normal table or cumulative-distribution function.

## Continuity correction

A binomial count takes whole-number values, while the normal curve is continuous. To include the entire probability bar at $k$, use a boundary halfway to the next integer. For example, $P(X\leq k)$ uses the normal area below $k+0.5$, and $P(X>k)$ uses the area above $k+0.5$. This adjustment is called a **continuity correction** [@pennStateNormalApproximationBinomial].

## Example: more than 210 tails

If a fair coin is tossed 400 times and $X$ counts tails, then $X\sim\operatorname{Binomial}(400,0.5)$. Its mean is $400(0.5)=200$ and its standard deviation is $\sqrt{400(0.5)(0.5)}=10$. Both the expected successes and failures equal 200, so the normal approximation is suitable.

“More than 210” means **211 or more** tails. Its complement is **at most 210** tails:

$$
P(X>210)=1-P(X\leq210).
$$

The empirical rule gives a quick estimate: 210 is one standard deviation above the mean, leaving about **16%** in the upper tail. For the more precise normal approximation, include the continuity correction and use the area above $210.5$:

$$
P(X>210)\approx P\left(Z>\frac{210.5-200}{10}\right)
=P(Z>1.05)\approx0.1469.
$$

So the corrected normal estimate is about **14.7%**. For comparison, the exact [Binomial Distribution](binomial-distribution.md) calculation is:

$$
P(X>210)=\sum_{k=211}^{400}\binom{400}{k}(0.5)^{400}
\approx0.146854=14.6854\%.
$$
