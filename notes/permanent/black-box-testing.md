---
title: Black-box Testing
date: 2021-04-04 00:00
tags:
  - AutomatedTesting
summary: Tests written without knowledge of software's internal structure
---

Black-box tests are tests that run without knowing anything about the internal structure of the software. They test what the software is expected to do from a business perspective.

White-box tests are the opposite of that. They test with a good understanding of the internal workings of the software.

Khorikov argues that since black box tests provide the best resistance to refactoring one of the [4 Pillars of Good Unit Tests](4 Pillars of Good Unit Tests.md), you should aim to mostly write tests in this style. Reserving white box testing for the analysis of your tests, for example, by utilising [test-coverage-metrics](test-coverage-metrics.md) to examine branch or code coverage. 

[@khorikovUnitTestingPrinciples2020] (pg. 90)