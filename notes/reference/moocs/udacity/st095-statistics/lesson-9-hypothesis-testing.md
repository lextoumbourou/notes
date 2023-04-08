---
title: "Lesson 9: Hypothesis Testing"
date: 2013-08-30 00:00
parent: st095-statistics
status: draft
modified: 2023-04-08 00:00
---

# Lesson 9: Hypothesis Testing

* Levels of likelihood
    * Probabilities statisticians have determined to be "likely" or "unlikely"
    * Alpha levels (very unlikely):
        * .05 (5%) (z-score = 1.64)
        * .01 (1%) (z-score = 2.33)
        * .001 (0.1%) (z-score = 3.1)
* Critical Regions
    * If probability is less than alpha level, will fall in the critical region in distribution:

    <img src="./images/critical_region.png"></img>

    * If we got a sample mean at 1.95, you might say "mean is significant at p < .05"
    * Above 2.33? "significant at p < .01"
* Two-tail critical values
    * When you need to consider both ends of the distribution
    * .05% needs to be divided in half: 2.5% * 2

    <img src="./images/two-tailed_test.png"></img>

    * Alpha levels (for two-tailed tests):
        * 0.05 - 1.65
        * 0.01 - 2.32
        * 0.001 - 3.08
* Null hypothesis vs alternative hypothesis
    * With a sample: can't prove null hypothesis is true, can only get evidence to reject it or fail to reject it

    <img src="./images/null_vs_alternative.png"></img>
* Rejecting the null
    * sample mean falls within the critical region
    * z-score of sample mean is greater than z-critical value
    * probability of obtainig the sample mean is less than the alpha level
* Decision errors
    * Type 1 error: reject the null and it's true
    * Type 2 error: retain the null and it's false

    <img src="./images/decision_errors.png"></img>
