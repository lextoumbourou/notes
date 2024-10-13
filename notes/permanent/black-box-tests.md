---
title: Black-box Tests
date: 2021-04-04 00:00
tags:
  - AutomatedTesting
summary: Tests written without knowledge of the software's internal structure
cover: /_media/black-box-testing-cover.png
---

**Black-box tests** are automated tests constructed without knowledge of the software's internal structure. They test what the software is expected to do from an end user's perspective - verifying just the output results given some inputs.

[White-box Tests](white-boxtests.md) are the opposite of that. They test with a good understanding of the internal workings of the software.

In [Unit Testing: Principles and Practices and Patterns](https://amzn.to/496VEy2), Khorikov argues that since black box tests provide the best resistance to refactoring one of the [4 Pillars of Good Unit Tests](4-pillars-of-good-unit-tests.md), you should aim to write tests in this style mostly. Reserve white box testing to analyse your tests, for example, by utilising [Test Coverage Metrics](test-coverage-metrics.md) to examine branch or code coverage.

---

## Further Reading

[Unit Testing: Principles, Practices and Patterns: Effective Testing Styles, Patterns, and Reliable Automation for Unit Testing, Mocking, and Integration Testing with Examples in C#](https://amzn.to/496VEy2)

![Cover for book Unit Testing: Principles, Practices and Patterns by Vladimir Khorikov](../_media/unit-testing-principles-practices-and-patterns.png)

This book is one of the most comprehensive guides to automated testing. I highly recommend it. Chapter 4 (pp. 87-90) compares black and box testing in-depth.
