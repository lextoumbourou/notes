---
title: Week 4 - Preventing Overfitting in Decision Trees
date: 2016-07-04 00:00
category: reference/moocs
parent: ml-classification
status: draft
---

## Overfitting in Decision Trees

### Review of overfitting

* Overfit = low training error, high true error.
* Occurs in overly complex models.

### Overfitting in decision trees

* As depth increases, training error moves to 0.
  * Why? When choosing feature to split on, you're picking lowest classification error: will eventually move to 0.
  * 0 training error is a big warning sign of overfitting.
* Want to find the depth with the lowest test error.

## Early stopping to avoid overfitting

### Principles of Occam's razor: Learning decision trees

* "Among competing hypotheses, the one with fewest assumptions should be selected", William of Occam, 13th Century.
* Example: have 2 symptoms to present to doctor, could either diagnose with 2 diseases for each sympton, or 1 disease that explains both symptoms.
* In decision trees: with 2 trees that have the same classification error, pick the simplest one.
* How to pick simpler trees:
  * Early stopping: stop learning when tree becomes too complex (eg the depth is above some threshold).
  * Pruning: simplify tree *after* learning algorithm terminates (generally the preferred approach).

### Early stopping in learning decision trees

* Condition 1: stop splitting are reach a max depth.
  * Use validation set for finding ideal max depth param.
* Condition 2: use classification error to limit depth of tree.
  * Stop recursing if classification error doesn't get better, eg, all choices of split at some leaf node don't get you a better classification error.
  * Generally add a magic param and stop if error doesn't decrease by said param.
  * Very useful in practise.
* Condition 3: stop if number of data points is low (below some threshold ``N_min``).
  * Should always implement this.

## Pruning Decision Trees

### Motivating pruning

* Problem with stopping condition 1:
  * How do you know the ideal max depth?
  * Can be fiddlying and imperfect trying to find it.
* Problem with stopping condition 2:
  * Short sighted: splits that seem useless (eg don't reduce classification error by much) can be followed by excellent splits.

### Pruning decision trees to avoid overfitting

* Firstly, you need a measure of tree complexity. Generally use number of leaves. ``L(T) = # number of leaf nodes in tree``
* Want to find good balance between low classification error and low complexity:
  * ``Total Cost C(T) = Error(T) + tuning_param * L(T)``
  * ``if tuning_param == 0:``
    * Standard decision tree example.
  * ``if tuning_param == float('inf'):``
    * Only lowest complexity trees will be choosen (classification error won't matter and will just choose majority class).
  * ``if tuning_param in range(0, float('inf')):``
    * Find balance between the 2.

### Tree pruning algorithm

* Roughly: walk through each node in the tree and throw away if cost function is lower without the split.

# Week 4: Handling Missing Data

## Basic strategies for handling missing data

### Challenges of missing data

* Missing data can impact at training time: dataset contains null values, or at prediction time: input to predict contains missing values.

### Strategy 1: Purification by skipping missing values

* Most common method.
* Just skip data points missing values.
  * Works if you have a "huge" dataset.
  * Can be problematic with small datasets where you need all the data.
* Could skip an entire feature if there are a lot of missing data points.
* Doesn't help if data is missing at prediction time.

### Strategy 2: Purification by inputing missing data

* Fill in missing values with a "calculated guess":
  * Categorical features should use mode of dataset (eg most commonly found value).
  * Numerical features use median (eg average value).
  * More advanced methods: expectation-maximization (EM) algorithm.
* Can result in systematic errors.

## Strategy 3: Modifying decision trees to handle missing data

### Modifying decision trees to handle missing data

* Add branch to decision tree when data is null.
* Creates more accurate predictions.
* Works for predictions and training.
* Need to modify training algorithm.

### Feature split selection with missing data

* When select a feature to split on also decide which branch to send the missing values down.
* Same principles as feature splitting: pick branch to assign missing values with lowest classification error.
