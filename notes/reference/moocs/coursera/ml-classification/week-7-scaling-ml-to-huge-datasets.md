---
title: Week 7 - Scaling ML to Huge Datasets
date: 2016-07-04 00:00
category: reference/moocs
parent: ml-classification
status: draft
---

## Gradient ascent won't scale to today's huge datasets

* Because gradient ascent requires scans over dataset while learning coefficients, it can be inefficient for large datasets.

## Timeline of scalable machine learning & stochastic gradient

* ML field has evolved with dataset size:
  * 90s: Small data = high model complexity required for accuracy.
  * 00s-10s: Larger datasets = simpler model. Lower model complexity mitigated by bigger datasets.
  * 10s-20s: Even larger datasets = high model complexity, new algorithms using GPUs, clusters, parallelism etc.
* Change to [Gradient Descent](../../../../permanent/gradient-descent.md) which updates coefficients as it iterates through the datasets.

## Scaling ML with stochastic gradient

### Stochastic gradient: learning one data point at a time

* Instead of summing all the data points to calculate the gradient, use a single data point to approximate it. Subsequent iterations can use a different, randomly chosen data point.
  * "Each update is cheaper, but you'll need more updates."

### Comparing gradient to stochastic gradient

* Stochastic
  * High sensivity to parameters
  * Achieves higher likelihood sooner, but is noisier.

## Understanding why stochastic gradient works

### Why would stochastic gradient ever work

* Basically: with standard gradient descent, you're always moving in the optimal direction towards the best coefficients. With stochastic, you are not neccessarily taking the optimal path but on average are making progress toward the goal.

### Convergence paths

* Stochastic gradient oscillates around optimal solution.
* Summary of stochastic vs gradient:
  * Gradient finds path of steepest ascent using contributes of all data points to find derivative.
  * Stochastic from path uses contributes of just 1 data point.
    * Sometimes decreases but on average will increase.
    * Has a noisy convergence path (oscillates around optimal solution).

## Stochastic gradient: Practical tricks

### Shuffle data before running gradient

* If data is impliciently sorted, can introduce bad practical performance.
  * Always mix up data before performing stochastic gradient ascent.

### Choosing the step size

* Very large step sizes can cause very bad behaviour.
  * Picking step size requires lots of trial and error with stochastic gradient descent.
* Ideal in stochastic: a step size that decreases per iteration

### Don't trust last coefficients

* Never full converges: will oscillate around optimal solution.
  * Take average of last learned coefficients, not the actual.

### Learning from batches of data

* Don't just learn from one data point, learn from a batch.
* Can make the convergence path smoother.

### Measuring convergence

* Estimate log-likihood with sliding window: use average of a bunch of data points to compute.

## Online learning: fitting models from streaming data

### The online learning task

* Batch learning = you have all the data when you start the learning process.
* Online learning = data streams in over time.
* Example:
  * Ad targeting: coefficients get updated as user clicks on certain ads.

### Using stochastic gradient for online learning

* Pros:
  * Models always up-to-date (always based on latest information in the world).
  * Lower computational cost.
  * Don't need to store all data.
* Cons:
  * Overall system is really complex.
* Most companies go with systems that update coefficients based on saved data at some scheduled interval.
