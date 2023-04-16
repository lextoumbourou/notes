---
title: "Week 3: Clustering with K-Means"
date: 2021-10-30 00:00
modified: 2023-04-08 00:00
status: draft
---

# Clustering with k-means

## Intro to clustering

### The goal of clustering

* Discover groups (clusters) of related articles.
* Learn user prefs: set of clusters read by user.
* Use feedback to learn user preferences over topics.

### An unsupervised task

* Firstly, example of supervisered classification problem: have set of labels, perform standard classification to assign labels to documents.
* Clustering: no labels provided, uncover cluster structure from input alone.
  * Input: docs as vectors ``X_i``
  * Output: cluster labels ``z_i``
* Cluster is defined by "center" & "shape/spread"
  * Assign observation ``x_i`` (doc) to cluster k (topic label) if:
    * Score under cluser k is higher than under others.
    * For simplicity, often define score as "distance to cluster center" (ignoring shape).

### Hope for unsupervised learning and some challenging cases

* 2 things that allow us to cluser:

  1. By definition of what a cluster is: what is the strucutre we're trying to extract from the data.
  2. Structure of the data itself.
    * Sometimes, the task is fairly easy: the data is easy clusterable.
    * Other cases, it's near impossible, depending on the structure of the data.

## Clustering via k-means

### The k-means algorithm

* The "bread and butter" clustering algorithm.
* Algorithm:
  1. Initial cluster centres somehow (guess that info is coming later?).
  2. Figure out distance from each datapoint and assign to its closest centre (the centre with the closest distance).

    $$z_i \leftarrow \text{arg min} ||\mu_j - \mathbf{x}_i||_2^2  $$

  3. Update position of cluster centre, based on data points assigned to it.
      * Basically, sum over all data points assigned to cluster and take the average.

      $$\mu_j = \frac{1}{n_j} \sum\limits_{i:z_i=j} \mathbf{x}_i $$

  4. Repeat step 2 and 3 until convergence.

### k-means as coordinate descent

* k-means is an example of coordinate descent.
* k-means converges to a local optimum, generally not a global optimum.
  * Cluster centre initialisation can result in vastly difference converges results (enter "smart initialisation"?).

### Smart initialisation via k-means++

* Algo:
  1. Choose first cluster centre uniformly at random data points.
  2. For each observation $$\mathbf{x} $$, compute distance $$d(\mathbf{x}) $$

to nearest cluster centre.

  3. Choose cluster centre from data points, with probability $$\mathbf{x} $$

being chosen proportional to $$d(\mathbf{x})^2 $$

     * In other words: find the next cluster centre that's far away from chosen cluster centre.
  4. Repeat Steps 2 and 3 until k centres have been chosen.

* Pros: high computational cost compared to randomly selecting.
* Cons: k-means converges more rapidly.

### Assessing the quality and choosing the number of clusters

* k-means objective: minimise the sum of squared distances in clusters across all clusters. Minimise this:

  $$\sum\limits_{j=1}^{k}\sum\limits_{i:z_i=j}||\mu_j - \mathbf{x}_i||^2_2 $$

  * Aka want low "cluster heterogeneity".
* As k increases, can end up with lower heterogeneity but also at risk of something akin to overfitting.
  * Extreme case thought experiment: k=N
    * heterogeneity = 0
    * can set each cluster centre equal to datapoint.
* How to choose k:
  * Want to trade off between low heterogeneity and useful clusters.
  * No right answer; depends on requirements.

## MapReduce for scaling k-means

### Motivating MapReduce

* 10B documents, single machine.
  * Init hashtable:

    ```
    count = {}
    for d in documents:
        for word in d:
             count[word] += 1
    ```

* MapReduce: distribute data and each word does a subset of them, then they're combined together.
* All counts for subset of words should go to the same machine.
* Map words to machines with a hash function:

  ```
  h(word index) -> machine index
  ```

### The general MapReduce abstraction

* Map:
  * data processed in parallel.
  * outputs ``(key, value)`` pairs.
    * "Value" can be any data type.

  ```
  def map(doc);
    for word in doc:
      emit(word, 1)
  ```

* Reduce:
  * Aggregate values for each key.
  * Order of operations cannot matter (must be commutative-associative).

  ```
  def reduce(word, counts_list):
    c = 0
    for i in counts_list
      c += counts_list[i]

    emit(word, c)
  ```

  * Also emits key value pairs.

### MapReduce - Execution overview

* Map phrase: emits key, values pairs across all data.
* Shuffle phrase: sort data and assign to machines for reducing.
* Reduce phrase: prepare data for output.
* Can use combiners to reduce data locally before the reduce step.
  * Word count example: do word count locally instead of emit ``word, 1`` for all data.
  * Works because reduce is "commutative-associatative"

### MapReducing 1 iteration of k-means

* Once you have initialised the cluster centre, for each data point you can figure out the distance to it (map step).
  * Map takes in set of cluster centers and a data point.
  * Emits the cluster assigned to and the data point.

    ```
    namedtuple('EmittedLabel', ['cluster_label', 'datapoint'])
    ```

* Then revise data points as mean of assigned observations (reduce step).
  * Count each data point and emit the average for each cluster.
* k-means is an iterative version of MapReduce.
  * Needs a bit more work than the standard MR approach.

## Summarizing clustering with k-means

### Other applications of clustering

* Grouping images - finding related images when searching for known images.
  * Handling search terms with multiple meanings.
* Clustering related seizures.
* Amazon products: discover product categories from purchase histories.
  * Discover groups of users.
* Discover similar neighbourhoods for estimating prices at similar neighbourhoods.
