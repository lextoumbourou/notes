---
title: Test Coverage Metrics
date: 2021-03-24 00:00
modified: 2021-05-27 00:00
tags:
  - AutomatedTesting
summary: "Metrics that answer the question : how much of our code is executed by our tests?"
---

Test coverage metrics answer the question: how much of our code is executed by our tests?

2 main metrics:

  $$\text{Code coverage (text coverage)}=\frac{\text{Lines of code executed}}{\text{Total number of lines}}$$

  $$\text{Branch coverage}=\frac{\text{Branches traversed}}{\text{Total number of branches}}$$

## Code coverage

The most commonly used metric.

The downside of using it that code can be "shuffled" to trick the metric into increasing. This can sometimes make code less readable.

## Branch coverage

Code can't be simplify shuffled without refactor, as branches will be examined whether one line or multiple.

The downside, which covers all code coverage metrics, is that tests without assertions will be counted as branch traversal.

[@khorikovUnitTestingPrinciples2020] (pg. 8-15)
