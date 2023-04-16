---
title: Week 3 - Decision Trees
date: 2016-07-04 00:00
category: reference/moocs
parent: ml-classification
status: draft
---

## Intuition behind decision trees

### Predicting loan defaults with decision trees

* Example of bank, looks at a bunch of factors when deciding whether to loan mony:
  * Credit history
  * Loan term
  * Income
* Can be represented as a "decision tree".

### Intuition behind decision trees

* Given input ``Xi = (credit = poor, income = high, term = 5 years)``, we traverse the decision tree ``T(Xi)`` to get the ``Y-hat i`` output.

### Task of learning decision trees from data

* Goal to learn decision tree that makes good predictions given input.
* Quality metric for algorithm: classification error:
  * Number of bad predictions / total predictions
    * 1 = very bad.
    * 0 = perfect.
* Learning the trees "perfectly" = NP-hard problem; there are exponentially large numbers of possible trees.
  * Need to approximate the tree; few simple algorithms that do that.
  * Course will cover: simple (greedy) algorithm.

## Learning decision trees

### Recursive greed algorithm (high-level outline)

* High-level algorithm:

  1. Start with empty tree (loans labelled as safe or risky)
  2. Then, split the tree on some feature (aka credit):
      * Excellent
      * Fair
      * Poor
    3. For each split, if all examples are excellent, then we can predict safe.
    4. Go to step 2 and continue spliting.

* Need to know when to stop recursing: covered in rest of module.

### Learning a decision stump

* Roughly: after splitting, pick the majority case as risky or safe (y-hat).

### Selecting best feature to split on

* Better split = one that gives lowest classification error.
  * Need to compare: splitting on nothing vs splitting on other features and compare classification error.

### When to stop recursing

* Criteria:
  * If all data in the split has just one class.
  * Used up all the features in your dataset.

## Using the learned decision tree

### Making predicitions with decision trees

* Once you have the tree, you just need to travese it for some input:

  ```
  func predict(tree_node, input):
    if tree_node.is_leaf:
        return tree_node.majority_class
    else:
        next_node = tree_node.get_next_node(input)   # Return input based on input data
        return predict(next_node, input)
  ```

### Multiclass classification with decision trees

* Majority class can still be used to predict result at leaf.
* Probability can be predicting by finding fraction for one class:

  ```
  safe = 3
  risky = 10
  danger = 18
  
  danger_prob = 18 / (3 + 10 + 18) = 0.58
  ```

### Threshold splits for continuous inputs

* With continuous inputs (eg income levels or age) you can't split on indiviual values of you'll get crazy overfitting (eg income at 30k is safe, but 34k is not etc), need to thresholding to come up with discreet "buckets" for your inputs.
  * Income buckets:

  ```
  * < 30k
  * >= 30k & < 100k
  * > 100k
  ```

### Picking the best threshold to split

* Roughly: choose the set of splits with the lowest classification error.
