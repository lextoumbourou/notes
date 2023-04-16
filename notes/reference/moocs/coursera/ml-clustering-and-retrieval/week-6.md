---
title: "Week 6: Recap"
date: 2021-10-30 00:00
modified: 2023-04-08 00:00
status: draft
---

# Week 6: Recap

## Module 1: recap

* 1-NN search
  * search over all articles to find closest article to query article.
* k-NN
  * same as 1-NN but return multiple articles.
* 2 critical elements to performance of NN:
  * how do you represent your docs?
  * how do you measure distance between two docs?
* TF-IDF doc representation
  * Emphasises words important locally but rare globally.
  * Term frequency = word count in doc.
  * Inverse doc frequency = ```log(# docs / (1 + # docs using word))```
* Scaled Euclidean distance
  * Allows for weighing certain features: eg weight words in title over body.
* Cosine similarity
  * Common metric for text data.
  * Inner product of 2 normalised vectors for each doc.
* Normalise or not?
  * Cosine similarity can ignore length of document.
  * Common comprimise: cap maximum word counts.
* Complexity of brute-force search
  * For each query point, need to iterate through entire dataset.
  * O(N) per 1-NN query.
  * O(N log k) per k-NN.
* KD-trees
  * Efficient data structure for NN.
  * Recursively partition the feature space.
  * Works by ensuring the distance to the points in your node is smaller than the distance to a lot of the other nodes.
  * Can prune more aggresively trading off speed for query accuracy by adding an alpha component to the prune condition:
    * Instead of pruning bounding box when > r, do it when > r / alpha.
* Limitations of KD-trees
  * Hard to implement.
  * Don't do well in high dimensions.
* Locality sensitive hashing
  * Throw down lines to partition feature space to create a series of "bin index" buckets..
  * Then store datapoints using binary represenation of query into a bucket.
  * Can search your bin and some neighbouring bin (by flipping one or more bits).

## Module 2: recap

* Introduced the concept of clustering.
* k-mean algorithm.
  * most widely used algorithm for clustering.
  * step by step:
    0. Init cluster centres.
    1. Assign observations to closest cluster centre.
    2. Revise cluster centres by averaging assigned observations.
    3. Repeat until convergence (cluster centres only changed by some alpha value).
* MapReduce can be used to parallelize k-means.

## Module 3: recap

* Mixture modules
  * A probabilistic clustering model.
  * Captures uncertainty in clustering.
  * Useful when k-means fails for the following reasons:
    * You have clusters of different sizes.
    * You have cluster overlap.
    * You have different shaped or oriented clusters.
* EM algorithm
  * E-step: estimate cluster responsibilities given params.
  * M-step: maximise likelihood over parameters given responsibility.
  * iterates between each step, with each improving the other.
* Relationship to k-means:
  * Gaussian mixture model basically becomes k-means when you have spherically symmetric clusters and variance parameter is 0 because datapoints get hard assigned to clusters.

## Module 4: Latent Dirichlet allocation

* Before presenting LDA, presented an alternate document clustering model using topic specific word distributions.
  * Docs are represented as BOW.
  * All words in document are assigned to cluster to determine overall document topics.
* LDA:
  * Every word scored under associated topic.
  * Determine distributions on topics in doc.
  * Has a topic proportion vector for each document.
  * Side note: can't remember how LDA is different from the initial alternate approach?
* Gibbs sampling
  1. Randomly assign topics for every word in document based doc topic proportions and topic vocab distributions (initially this would be assigned random or through some other means -- like k-means :) ).
  2. Randomly reassign topic proportions based on assignments in current doc.
  3. Repeat for all docs.

## Hierarchical clustering and clustering for time series sementation

### Why hierarchical clustering

* Avoids having to set number of clusters beforehand (however, trade off is other params need to be fixed)
* "Dendrograms" help visualise different cluster granularities without having to rerun.
* Most algo let user choose any distance metric (where k-means requires Euclidean distance).

### Divisive clustering

* Practise of breaking up a cluster into multiple clusters.
* Example given:
  * Break up Wikipedia article into athletes and non-athletes.
  * Break up those articles into the different sports for athletes and other types for non-athletes.
* Choices you need to make:
  * Algorithm to recurse?
  * How many clusters per split?
  * When to split vs stop
    * Max cluster size.
    * Max cluster radius.
    * Specified # clusters.

### Agglomerative clustering

* Example used: "single linkage"
* Algo:
  1. Init each point to be its own cluster.
  2. Define distance using pairwise distance function & linkage criteria.
    * Find 2 points that are closest from each cluster and define that as the distance between clusters.
  3. Merge the two closest clusters that have the min distance.
  4. Repeat step 3 until all points are in 1 cluster.

### Dendrogram

* A way to represent the results of hierarchical clustering.
* No idea what the point of this slide is?

### Agglomerative clustering details

* What are the choices you have to make in agglomerative clustering?
  * What distance metric?
  * What linkage function?
  * Where and how to cut dendrogram?
* Cutting dendrogram
  * For visualization, small # clusters is prefered.
  * Outlier detection, can cut based on:
    * Distance threshold.
    * Inconsistency coefficient (don't get it - does it matter?).
* Computational considerations
  * Compute distances between all points: brute force is ``O(N^2 * log(N))``
  * Smart implementations use triangle inequality to rule out candidate pairs.
  * Best known algo is ``O(N^2)``.
* Statistical issues
  * Chaining: distance points can be very small but with a lot of points, you can eventually have a big disparity between two extremes of distance in a cluster.
  * Other linkage functions can be more robust but restrict the shape of clusters:
    * Complete linkage: max pairwise distance between clusters.
    * Ward criterion: min within-cluster variance at each merge.

### Hidden Markov models

* Clustering so far only looked at under ordered data ie when datapoint was added not important.
* What about if the data is time series data?
  * Clustering useful to provide pattern amongst the time ordered datapoints.
* Hidden Markow model (HMM)
  * Similar to mixture model.
  * Every observation is associated with cluster.
  * Each cluster has a distribution over observed values.
  * Difference: probability of cluster assignment depends on previous cluster assignment.
* Inference in HMMs
  * Learn MLE of HMM parameters using EM algorithm: Baum Welch.
  * Infer MLE of state sequence given fixed model parameters using dynamic programming: Viterbi algorithm.
  * Infer soft assignments of state sequence using dynamic programming: forward-backward algorithm.
