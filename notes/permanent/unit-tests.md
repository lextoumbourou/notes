---
title: Unit tests
date: 2021-03-27 00:00
tags:
  - AutomatedTesting 
summary: An automated test which verifies a single unit of behaviour, runs quickly and in isolation
---

The definition of a unit test, according to Khorikov, is an automated test which meets the following criteria:

* Verifies a single unit of behaviour
* Executes quickly
* Runs in isolation from other tests.

Two schools of thought on unit testing: [London (mockist)](London%20(mockist)) and [Classical (Detroit)](Classical%20(Detroit)), differ mainly on the intepretation of the third point. The former views isolation as being from all dependancies: use of mocks should ensure only one class is tested at a time whereas the classic refers to isolation between tests and their ability to parallelise.

On the other hand, an [integration test](integration%20test) is simply any test which can't satisfy this criteria.

[@khorikovUnitTestingPrinciples2020] *(chapter 2)*
