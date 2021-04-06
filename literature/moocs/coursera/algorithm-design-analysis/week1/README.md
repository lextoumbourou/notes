# Week 1

## Introduction

### Integer Multiplication

* Input: 2 n-digit numbers, x and y.
* Output: the product, x * y.
* Assess performance: number of basic operations performed as a function of input size (n).

* Primary school algorithm:
  * "Correct" - will eventually terminate with correct answer.
  * Requires at most 2n operations per row (multiple first number by each top number and potentionally carry the remainder).

### Karatsuba Multiplication

```
x = 5678
y = 1234

a = 56
b = 78
c = 12
d = 34
```

* Step 1: Compute a * c  = ``56 * 12 = 672``
* Step 2: Compute b * d  = ``78 * 34 = 2652``
* Step 3: Compute (a + b)(c + d) = ``(56 + 78) * (12 + 34) = 6164``
* Step 4: Compute 3 - 2 - 1 = ``6164 - 2652 - 672 = 2840``
* Step 5: Start with 1 and pad with 4 000s. Take 2 no padding. Take 4 with 3 0 padding = ``6720000 + 2652 + 284000 = 7006652``

* Not expected to have intuition for it: just for appreciating that there are different algorithms for stuff.

#### A recursive algorithm

Firstly, rewrite x and y as follows:

```
x = (10 ^ (n/2) * a) + b == x = 5600 + 78
y = (10 ^ (n / 2) * c + d) == y = 1200 + 34
```

Then

```
x * y = (10 ^ (n/2) * a + b) * (10 ^ (n / 2) * c + d)
      = 10^n * ac + 10^(n/2) * (ad + bc) + bd
```

Compute ``ac, ad, bc, bd`` then compute in the straight-forward way.

### Merge Sort: Motivation and Example

* Why study merge sort?
  * 60 - 70 years old, but still used widely.
  * Performance improvements over selection, insertion and bubble sort.
    * All have quadradic times.
  * Calibrate your preparation.
* Overview:

  * Take initial array and split in half.
  * Sort each half.
  * Merge using the following algorithm:

    ```
    let A = first_half
    let B = second_half
    let C = array of input size.
    i = 1
    j = 1
    for k in range(len(C)):
        if A[i] < B[j]:
            C[k] = A[i]
            i++ 
        else:
            C[k] = B[j]
            j++ 
    ```
  * Running time of merge = ``4n + 2`` (4 operations for size of input + 2 initialising)
    * Simplify to ``6n`` because n is always at least 1.
  * Claim: merge sort requires: ``<= 6n * log(n, 2) + 6n``
    * Never explained where the first 6n comes from.
  * Logarithm intuition: how many division by the base number to get to 1 or less (aka identity function).
    * ``log(n, 2) = 32 / 2 -> 16 / 2 -> 8 / 2 -> 4 / 2 -> 2 / 2 = 5``

### Merge Sort: Analysis

* Can represent each layer of recursion as a tree:

    ```
    level 0         [5, 6, 8, 1, 9, 3, 4, 2]
                         /         \
    level 1         [5, 6, 8, 1]  [9, 3, 4, 2]
                        /   \         /      \
    level 2         [5, 6] [8, 1]    [9, 3]   [4, 2]
                      / \    /  \     /  \    /   \
    level 3         [5] [6] [8] [1]  [9] [3] [4] [2]
    ```

  * We know the merge requires ``6n`` operations roughly.
  * We know that we'll need ``log 2 n`` levels.

  * At each level ``j``, we know there'll be ``2^j`` nodes:

      * j = 0, ``2 ** 0 == 1``
      * j = 1, ``2 ** 1 == 2``
      * j = 2, ``2 ** 2 == 4``
      * j = 3, ``2 ** 3 == 8``

  * At each level ``j`` we know the node sizes will be ``n / (2^j)``

      * j = 0, ``8 / (2 ** j) == 8 / 1 == 8``
      * j = 1, ``8 / (2 ** j) == 8 / 2 == 4``
      * j = 2, ``8 / (2 ** j) == 8 / 4 == 2``
      * j = 3, ``8 / (2 ** j) == 8 / 8 == 1``

  * So, each level has a run time of: ``2^j * 6(n / 2^j)``
  * Then, the ``2^j``s are cancelled out.
  * So, each level has a run time of ``6n`` regardless of input size.
  * So, we can represent the run time of all levels as ``6n * log2n`` + the base case:
    ``6n * log2n + 6n``

### Guiding Principles for Analysis of Algorithms

* 3 guiding principles:

  1. Look at "worst-case analysis".
    * Doesn't require domain specific knowledge.

  2. Ignore constant times. 
    * Generally becomes irrelevant as problem size increases.

  3. Asymptotic analysis: focus on *laaarge* input sizes.

* Fast algorithm == "worst case running time grows slowly with input size"

## Asymptotic Analysis

* Motivation: vocabulary for design and analysis of algorithms.
* High-level idea: suppress constant factors and lower-order terms.

### Big-Oh Notation
