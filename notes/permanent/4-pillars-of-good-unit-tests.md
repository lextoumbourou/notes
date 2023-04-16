---
title: 4 Pillars of Good Unit Tests
date: 2021-04-04 00:00
summary: Khorikov's 4 pillars of good unit tests
tags:
  - AutomatedTesting
---

Khorikov describes 4 pillars of a good unit test:

1. Protection against regressions (bugs)
2. Resistant to refactoring
3. Fast feedback
4. Maintainability

He claims that the first 3 are mutually exclusive and provide a 2-out-of-3 trade off similar to the [CAP Theorem](cap-theorem.md) trade off. Since maintainability is not mutually exclusive, it should always be maximised. However, some types of tests lend themselves to more maintainabiity: e2e tests for example require a lot more code than a unit test.

Resistance to refactoring should be maximised always, as it is either binary: it is or it isn't. The trade off is then between resistance to refactoring or fast feedback.

An extreme example of this is an e2e test. It is extremely resistant to refactoring but not usually very fast. Another example of this is a heavily mocked method, it is very fast but may not provide good protection against regressions, since the mock can easily make assumptions that don't exist. This example would also fail to score high in the resistant to refactoring pillar.

[@khorikovUnitTestingPrinciples2020] *(Chapter 4)*
