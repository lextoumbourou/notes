---
title: "Lesson 8: Estimation"
date: 2013-08-30 00:00
parent: st095-statistics
status: draft
modified: 2023-04-08 00:00
---

# Lesson 8: Estimation

* Confidence interval:
    * Take a sample and get the mean
    * What is the probabilty that the mean weight of the *population* is within a range of the sample mean
* z-scores that bound 95% of a normal distribution
    * -1.96
    * 1.96

<img src="./images/z-score_bounds.png"></img>

* 95% confidence interval for a sample mean
    * ```sample_mean - SE < mean < sample_mean + SE```
* General idea:
    * The bigger the sample, the more confident one can be that mean derived from it matches the population mean
    * SE = population_standard_deviation / sqrt(sample_size) = a smaller value as the sample size gets large
* Critical values of z
    * 2.33 = 98%
    * 1.96 = 95%
* Practical use of Confidence Interval
    * Test a change (treatment) and collect data (dependent variables) from a population
    * Use it to determine where it lies on a sample mean distribition (z-score using SE)
    * Then, use z-table to calculate probability of getting that
