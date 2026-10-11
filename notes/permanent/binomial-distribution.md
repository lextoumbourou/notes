---
title: Binomial Distribution
date: 2026-10-10 20:49
modified: 2026-10-10 20:50
status: draft
summary: The probability model for counting successes across independent trials.
tags:
  - Statistics
  - Probability
---

The **binomial distribution** describes the number of successes $X$ in a fixed number $n$ of independent trials. Each trial has two outcomes, success or failure, and the probability $p$ of success stays the same. We write $X\sim\operatorname{Binomial}(n,p)$ [@nistBinomialDistribution].

“Success” means the outcome we choose to count. In a game with several kinds of prize, we could count “small prize” as success and group every other outcome as failure, provided each game is independent and the chance of a small prize stays fixed.

The probability of exactly $k$ successes is:

$$
P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}.
$$

The [Combination](combination.md) $\binom{n}{k}$ counts which $k$ trials were successes. The remaining factor is the probability of any one such sequence. The distribution has mean $np$ and [Standard Deviation](standard-deviation.md) $\sqrt{np(1-p)}$ [@nistBinomialDistribution].

For example, if a fair coin is tossed three times and a tail counts as success, then $n=3$, $p=0.5$, and:

$$
P(X=2)=\binom{3}{2}(0.5)^2(0.5)^1=\frac{3}{8}.
$$

When $n$ is large enough, the [Normal Approximation to the Binomial Distribution](normal-approximation-to-the-binomial-distribution.md) can estimate probabilities for ranges of counts. This statistical distribution is distinct from a [Binomial](binomial.md) in algebra, which is an expression with two terms.
