---
title: "Week 2: Nearest Neighbour Search"
date: 2021-10-30 00:00
modified: 2023-04-08 00:00
status: draft
---

# Week 2: Nearest Neighbour Search

## Introduction to nearest neighbor search and algorithms

## Retrieval as k-nearest neighbour search

* Document retrieval
  * Goal: retrieve an article that might be relevant to user based on what they're currently reading.
  * Take article, find all articles that are similar to current article, then sort based on closeness to current article.

## 1-NN algorithm

* Input: Query article
         Corpus of documents (N docs)
* Output: Most similar article.
  * Formally: ``Xnn = min_distance(Xq, Xi)``
  * In code:

    ```
    def one_nn(corpus, query_doc):

      """One nearest neighbour."""

      closest_distance = float('inf'), closest_doc = None
      for i in range(len(corpus)):
        distance = distance(document[i], query_doc)
        if distance < closest_distance:
          closest_distance = distance
          closest_doc = document[i]

      return closest_doc
    ```

    (Obviously the interesting part is going to be the distance calculation).

## k-NN algorithm

* Same as 1-NN but return a set of documents that are closest to your document.
  * Use queue to shift further article when a closer one comes along.

## The importance of data representations and distance metrics

### Document representation

* Bag of words model:
  * Count number of instances of words
  * Ignore order
* Common words like ``the``, ``and`` can dominate word counts; enter TF-IDF
  * Emphasises words that are common locally but are rare globally.
  * Term frequency = word counts in document
  * Inverse doc freq: ``log(# docs / 1 + # docs using word)``
  * ``tf * idf``

### Distance metrics: Euclidean and scaled Euclidean

* In 1D, Euclidean distance: ``distance(x_i, x_q) = abs(x_i - x_q)``
  * Would only work if you had 1 word in your vocabulary.
* In multi dimensions:
  * many interesting distance functions
  * can weight dimensions differently
* Reasons for weighting different features:
  * Some features more relevant than others.
    * Example: weight the title more than the body
  * Some features vary more than others (??)
    * Specify weights as function of feature spread
      * for feature j: ``1 / max_i(x_i[j]) - min_i(x_i[j])``
* Scaled Euclidean distance:

     $$distance(\mathbf{x}_i, \mathbf{x}_q) = \sqrt{a_1(\mathbf{x}_i[1] - \mathbf{x}_q[1])^2 + ... + a_d(\mathbf{x}_i[d] - \mathbf{x}_q[d])^2} $$

	* Add up the distance for each scaled feature, between the query doc and some other doc and take the square root.
	* Can turn off features by setting weight to 0.

### Writing (scaled) Euclidean distance using (weighted) inner products

* Can rewrite the non-scaled Euclidean distance using linear algebra. Example:
    * $$distance(\mathbf{x}_i, \mathbf{x}_q) = \sqrt{(\mathbf{x}_i[1] - \mathbf{x}_q[1])^2 + ... + (\mathbf{x}_i[d] - \mathbf{x}_q[d])^2}  $$

can be rewritten as:

	* $$distance(\mathbf{x}_i, \mathbf{x}_q) = \sqrt{(\mathbf{x}_i - \mathbf{x}_q)^T(\mathbf{x}_i - \mathbf{x}_q)} $$

	* Basically: taking the dot product of the two feature distance vectors is equivalent to squaring then adding the results.
* To add the feature weights, you can add a diagonal matrix of features (note: not clear on the diagonal matrix part):
	* $$distance(\mathbf{x}_i, \mathbf{x}_q) = \sqrt{(\mathbf{x}_i - \mathbf{x}_q)^T\color{blue}{\mathbf{A}}(\mathbf{x}_i - \mathbf{x}_q)} $$

### Distance metrics: Cosine similarity

* Another inner product measure: multiple features in one document vs another then add up results:
	* $$\sum\limits_{j=1}^{d}\mathbf{x}_i[j]\mathbf{x}_q[j] $$
	* Higher is better (map overlap). Lowest possible = 0.
* Cosine similarity: same as above, however, you divide the result by the normalised vector of features:
	* $$\frac{\sum\limits_{j=1}^{d}\mathbf{x}_i[j]\mathbf{x}_q[j]}{\sqrt{\sum\limits_{j=1}^{d}(\mathbf{x}_i[j])}\sqrt{\sum\limits_{j=1}^{d}(\mathbf{x}_q[j])}} $$
	* Can be rewritten like: $$\frac{\mathbf{x}_i^T\mathbf{x}_q}{||\mathbf{x}_i|| ||\mathbf{x}_q|| } $$
* Cosine distance = 1 - cosine similarity
	* Not a proper distance metric because "the triangle equality doesn't hold".
	* Efficient to compute for sparse vectors because you only need to compute non-zero elements.
	* Very similar documents would result in a score close to 1. $$cos(0) \approx 1 $$
	* With only positive features, similarity would be between 0 and 1.

### Normalise or not?

* When you don't normalise, a document twice the size of the other would result in a much larger similarity score.
* Normalisation is not always desired: normalising can make dissimilar objects appear more similar, especially with smaller documents.
    * Compromise: cap maximum word counts.
* Other distance metrics: Mahalanobi, rank-based, correlation-based, Manhatten .
* Combining distance metrics:
    1. Text of document (cosine similarity)
    2. \# of reads of the doc (euclidean distance)
    3. Add together with user-specified weights

## Scaling up k-NN search using KD-trees

### Complexity of brute force search

* Given a given query point, need to scan through all others.
  * 1-NN query = O(N)
  * k-NN = O(N log k) (log k because can efficient use a priority queue)

### KD-tree representation

* Allows for efficiently pruning search space (works in many but not all cases).
* Start with list of d-dimension, choose a feature and a split feature. Everything on one side falls within one side of the threshold etc.
  * Then, in the partitioned space, choose another feature and split threshold etc.
  * At each node in tree, you store 3 things:
    * the dimension you split on.
    * the split value.
    * bounding box that is as small as possible while containing points (??).
* Use heuristics to make splitting decisions:
  * Dimensions to split on: widest or alternate.
  * Value to split on: median (or centre of box, ignoring data in box)
  * When to stop: fewer than m pts left or box hits minimum width.

### NN search with KD-trees

* NN search algo:
  1. Start by exploring leaf node for query point.
  2. Compute distance to each other point at leaf node.
  3. Backtrack and try other branches at each node visited.
* Can use bounding box of each node to prune parts of the tree that def won't contain NN.

### Complexity of NN search with KD-trees

* For (nearly) balanced binary trees:
  * Construction
    * Size: 2N - 1 nodes if 1 datapoint at each leaf.
    * Depth: O(log N)
    * Median + send points left right: O(N) at every level of the tree
    * Construction time: O(N log N)
  * 1-NN query
    * Traverse down tree to starting point: O(log N)
    * Maximum backtrack and traverse: O(N) worst case.
    * Complexity range: O(log N) -> O(N)

### Visualizing scaling behavior of KD-trees

* Ask for nearest neighbor to each doc:
  * Brute force 1-NN: ``O(N^2)``
  * kd-trees: ``O(N log N) -> O(N^2)``

### Approximate k-NN search using KD-trees

* Idea: the best nearest neighbor is not always necessary, near-enough can suffice.
* Before: prune tree when distance to bounding box is > ``distance found so far``.
* Now: prune when distance to bounding box > r / alpha.
  * Alpha = some number greater than 1.
  * Produces a smaller bounding box to prune more aggressively.
  * Could potentially prune closer neighbors, but saves a lot of searching.
* Lots of variants of kd-trees
* High-dimension spaces can be hard to put into KD-trees.

## Locality sensitive hashing for approximate NN search

### Limitations of KD-trees

* Limitations:
  * Not simple to implement.
  * Problems with high-dimensional data.
* KD-trees in high dimensions
  * Low chance of having data points close to query point.
    * With many dimensions, the splits could be all over the place.
  * Once you find your nearby point, the search radius has to intersect many hypercubes.
  * Hard to prune a lot of nodes.
* Moving away from exact NN search
  * Don't find exact neighbour; okay for many applications.
  * Fous on method that provide probabilibites of NN.

### LSH as an alternative to KD-trees

* LSH = locality-sensitive hashing
  * Partion data into bins by calculating sign of score then putting -1 into ``0`` bin and +1 into ``1`` binthen putting -1 into ``0`` bin and +1 into ``1`` bin.
  * Then use hash table with bin index as key and list of query points in bin as value.

    ```
    {
      0: {1, 5, 6, 9},
      1: {3, 10, 23, 7}
    }
    ```

  * Doesn't always return an exact nn.

### Using random lines to partition points

* 3 potential issues with LSH approach

  1. Challenge to find good parition line where goal is to have points in same bin that have close cosign similarity.
    * Consider random line. Seems like a bad idea but there is actually a good probability that close points would end up in the same bin.
  2. Poor quality solution: points close together get split into separate bins
  3. Large computational cost: bins might contain many points, so still searching over large set for each NN query.
      * With one line, you are only really halving the search space.

### Defining more bins

* Idea: add multiple lines. Instead of using single binary value as index, use multiple binary values.

  ```
  {
      (0, 0, 0): {1, 7},
      (0, 0, 1): {2, 6},
      (0, 1, 1): {3, 9}
  }
  ```

  * Somehow you get different scores for each line value and can use that to calculate the values in the index tuple (don't really get how that works...).

### Searching neighbouring bins

* The more lines you have, the more likely the nearest neighbour will not be in the same bin.
  * Solution: search more bins for nn.
* To find next closest bins, flip 1 bit: ``(0, 0, 0)`` -> ``(0, 1, 0)``
* Could even consider flipping 2 bits: ``(0, 0, 0)`` -> ``(1, 1, 0)``
* Algorithm: keep searching bins until computative cost is maxed out or quality of NN is good enough.
  * Lets you control computation complexity vs accuracy tradeoff.

### LSH in higher dimensions

* Cost of binning in d-dimensions: requires d multiplications to compute.

  ```
  Score(x) = v1 * #awesome + v2 * #awful + v3 * #great
  ```

  * Cost is required to be computed once, then is efficient for future queries.
  * For sparse datasets, lots of multiplication can be skipped.

### Improving efficiency through multiple tables

* Basic idea: create multiple hash tables with randomly generated lines and search them all for a single point.
  * Probability of finding NN *mostly* higher in the multiple table case.
