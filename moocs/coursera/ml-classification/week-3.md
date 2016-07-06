# Week 3

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
