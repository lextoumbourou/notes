---
title: Discrete Mathematics with Applications - Chapter 9
date: 2023-03-06 00:00
category: reference/books
status: draft
---

## Chapter 9

## 9.5 Binary Trees

* Most important of the m-ary trees and have wide range of applications.
    * Model tournaments, for example.
* Supports representation and evaluation of algebraic expressions.
* Binary tree is 2-ary:
    * Every internal vertex has at most 2 children:
        * Elder is left, and other is right.
    * Subtree at left child $v$ is the **left subtree** rooted in $v$
    * Subtree at right child $w$ is the **right subtree** rooted in $w$
    ![chapter-9-subtree-example](../../../journal/_media/chapter-9-subtree-example.png)

### 9.6 Binary Search Trees

* Binary trees that contain items of the same kind are called: "homogeneous trees"
* [Binary Search Tree](Binary%20Search%20Tree)
    * A homogeneous binary tree where every item of the left subtree of each vertex $v$ is less than $v$, and every item on the right is greater than $v$.
    * ![chapter-9-binary-search-tree](../../../journal/_media/chapter-9-binary-search-tree.png)
    * Binary search trees worst case is h + .1 comparisons (although according to the answer it's actually h)

### Exercise 9.6

Construct a binary search tree for each set.

5. 8, 5, 2, 3, 13, 21

![chapter-9-q5](../../../journal/_media/chapter-9-q5.png)

6. 5, 2, 13, 17, 3, 11

![chapter-9-search-tree-q6](../../../journal/_media/chapter-9-search-tree-q6.png)

15. Maximum comparisons to locate an item in #5 = 3
16. Maximum comparisons to locate an item in #6 = 3
