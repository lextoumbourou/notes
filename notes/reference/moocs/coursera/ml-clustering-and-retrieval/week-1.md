---
title: "Week 1 - Clustering and Retrieval"
date: 2021-10-30 00:00
modified: 2023-04-08 00:00
status: draft
---

# Week 1: Welcome

* What is retrieval?
  * Search for related items.
  * Input = features for query point + features of all other datapoints.
  * Output = "nearest" point or set of points to query.
  * Examples = find "nearest neighbour" of a set of articles (articles like this).
* What is clustering?
  * Discover groups of similar inputs.
  * Examples: discover groups of people with similar interests for ad targetting.

## Course overview

* Models:
  * Nearest neighbors.
  * Clustering.
  * Mixture of gaussians.
  * Latent dirichlet allocation.
* Algorithms:
  * KD-trees.
  * Locaility sensitive hashing.
  * k-means.
  * MapReduce.
  * Expectation Maximization.
  * Gibbs sampling.
* Core ML:
  * Distance metrics.
  * Approximation algorithms.
  * Unsupervised learning.
  * Probabilistic modeling.
  * Data parallel problems.
  * Bayesian inference.

## Module-by-module topics covered

* Module 1: nearest neighbour search
  * Compute distances to other documents and return closest.
  * Brute force can be slow, KD-trees are a fast, efficient and approximate NN search.
  * LSH: more effective than KD-trees in higher dimensions.
* Module 2: k-means and MapReduce
  * Discover clusters of related documents with k-means.
    * Aims to minimize sum of squared distances to cluster centers.
  * Scaling up k-means with MapReduce.
* Module 3: Mixture Models
  * Probabilistic clustering models.
    * Captures uncertainty in clustering.
  * Helps learn user topic preferences.
* Module 4: Latent Dirichlet Allocation
  * Allows for mixed membership: document can belong to multiple topics.
  * Document is assigned a probability for each topic.
