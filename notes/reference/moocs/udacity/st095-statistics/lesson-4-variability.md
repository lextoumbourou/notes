---
title: "Lesson 4: Variability"
date: 2013-08-30 00:00
parent: st095-statistics
status: draft
modified: 2023-04-08 00:00
---

# Lesson 4: Variability

* Quanify spread == work out "spreadoutness" of distribution
* Outliers increase the variability of data
* Dealing with outliers:
    * Cut off tail of data - lower 25% and upper 25%
    <img src="./images/q1_q3.png"></img>
        * Called the 'Interquartile Range' or IQR
* IQR
    * 50% of data falls within IQR
    * It's not affected by every value in dataset like outliers
* Outlier formula:
    * ```outlier < (q1 - 1.5 * iqr)```
    * ```outlier > (q3 + 1.5 * iqr)```
* Boxplots (aka box-and-whisper plots)
    <img src="./images/boxplots.png"></img>
* Variance:
    * Mean of squared deviations

```
sum(each deviation_from_the_mean**2) / sample_count
```

* Standard deviation (lower-case sigma)
    * ```sqrt(variance)```
<img src="./images/std_dev.png"></img>
* Properties of std dev
    * ~68% of data falls within 1 std devs of the mean in either direction
    * ~95% of data falls within 2 std devs of mean in either direction
<img src="./images/std_dev_distribution.png"></img>
* Bessel's correction
    * Samples tend to be values in the middle of population
    * Variability in sample will be less than in population
    * Instead of dividing by n, divide by n-1 when calculating variance and std dev of a sample
    * Called 'sample standard deviation'
<img src="./images/sample_std_dev.png"></img>
