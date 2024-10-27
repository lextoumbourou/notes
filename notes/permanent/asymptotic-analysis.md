---
title: Asymptotic Analysis
date: 2024-01-18 00:00
modified: 2024-01-18 00:00
status: draft
---

**Asymptotic Analysis** of a function, $f(n)$, tells us what happens to the function as $n$ grows larger. The term "asymptotic" refers to the behavior or value of a function as its input approaches a particular value or limit - in this case, as the input size approaches infinity.

## [Complexity Analysis](complexity-analysis.md)

The complexity of an algorithm tells us how many resources are required to complete an algorithm.

### [Time Complexity](time-complexity.md)

How many steps is required to complete the algorithm?

### [Space Complexity](space-complexity.md)

How much memory is required to complete the algorithm?

## [Big-O Notation](big-o-notation.md)

Big-O finds the upper bound of a function's asympototic growth.

When comparing a function's asymptotic growth, we can look at only the parts that grow the fastest. Big-O notation describes just the fastest growing factor of a function, to simplify the comparison of algorithms.

For example,
* $f(n) = 2^{n} + 3n$ in Big-O notation is $O(2^n)$
* $g(n) = 1000n^2 + n$ is $O(n^2)$

## [Worst-Case Time Complexity](worst-case-time-complexity.md)

Since the $O$ for an algorithm will vary depending on inputs, worst-case time complexity tells us what how long the model will take with the worst possible input.
