---
title: Normal Distribution
date: 2026-10-05 16:07
modified: 2026-10-05 16:07
status: draft
summary: How the normal curve connects standard deviations, probabilities and percentiles.
tags:
  - Statistics
  - Probability
---

Some datasets have histograms that look bell-shaped. Heights are a common example: most measurements are near the middle, with fewer at either end. We say these data **approximately follow the normal curve**. Other datasets have histograms that look different. Incomes and house prices, for example, are often skewed rather than bell-shaped.

The normal curve is a smooth model. The histogram of real measurements does not have to trace it exactly. The **area under the curve** represents probability, and the total area is 1 [@nistNormalData].

## The empirical rule

If data approximately follow the normal curve, the **empirical rule** is a useful rule of thumb [@nistNormalData]:

- About **68%** of the data fall within one [Standard Deviation](standard-deviation.md) of the mean. That is roughly two-thirds.
- About **95%** fall within two standard deviations.
- About **99.7%** fall within three standard deviations.

The curve is symmetric, so the area left over is split between the two sides. After one standard deviation, about **16%** is in each tail. After two standard deviations, about **2.5%** is in each tail.

For example, suppose fathers' heights have a sample mean of $\bar{x}=68.3$ inches and a sample standard deviation of $s=1.8$ inches. The empirical rule suggests that about 95% of their heights fall between $68.3-2(1.8)=64.7$ inches and $68.3+2(1.8)=71.9$ inches.

## Standardising data

To standardise a height, subtract the sample mean and divide by the sample standard deviation:

$$
z=\frac{\text{height}-\bar{x}}{s}.
$$

The result is a **z-score**. It has no units: in this example, inches in the numerator and denominator cancel. A z-score of $1.5$ means a height is 1.5 standard deviations above the average. A z-score of $-0.52$ means it is 0.52 standard deviations below the average.

When we standardise every value using the same sample mean and standard deviation, the resulting values have a mean of 0 and a standard deviation of 1. If the original values follow a normal distribution, the standardised values follow the **standard normal distribution**. Its curve is given by [@nistNormalData]:

$$
\phi(z)=\frac{1}{\sqrt{2\pi}}e^{-z^2/2}.
$$

The height of that curve is a density. To find a probability, we look at the **area** under it.

## Computing a percentile

What is the 30th percentile of the fathers' heights in the example? It is the height below which about 30% of the values fall. A standard normal table gives $z\approx-0.52$ for an area of 30% to the left.

We can undo the standardisation to turn the z-score back into a height:

$$
\text{height}=\bar{x}+zs=68.3+(-0.52)(1.8)\approx67.4\text{ inches}.
$$

So the normal model estimates the 30th percentile at about **67.4 inches**.
