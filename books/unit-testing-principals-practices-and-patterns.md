---
Link: https://www.amazon.com/Unit-Testing-Principles-Practices-Patterns/dp/1617296279
Title: Unit Testing Principles, Practices, and Patterns
Author: [[Vladimir Khorikov]]
Type: #book
tags: #GameDesign
aliases: ["# Unit Testing Principles, Practices, and Patterns"]
---

# Unit Testing Principles, Practices and Patterns

## 1. The goal of unit testing

* State of unit testing:
    * Unit tests are a now an industry standard: very few people do not think they're a good idea.
    * However, only some unit test practices actually yields desired outcomes, others may be making things worse.
    * Testing goal: strive to achieving best return for least effort.
    * Ratio of prod code to tests between 1:1 - 1:3 (1 line of prod code for 3 lines of test) and up to 1:10
* Software entropy: phenomenon of quickly decreasing development speed as project increases.
* Goal of unit testing:
    * Sustainable growth of software project.
        * Easy to grow project from scratch much more difficult as entropy increases.
    * Tests provide insurance against many types of regressions.
    * Though test practices can lead to better design, it's not the primary goal of unit testing
* Downside of tests: big upfront investment.
* What makes a good or bad test?
    * Does it reduce:
        * Amount of test refactoring required when code is refactored
        * Speed of running test when code changes
        * Amount of false alarms from test running.
        * Time spent reading the tests
* Tests are code and code is a liability not an asset.
    * Tests are vulnerable to bugs and require maintenance.
* Use of coverage metrics for assessing test quality
    * Coverage metrics should how much code was tested as a percentage from 0 to 100%.
    * Types of coverage metric:
        * Code coverage: ```code coverage = lines of code run during test / total lines``
            * Easy to game: you can something reduces branches to single line to fool the metric
        * Branch coverage: ```branches traversed / total branches```
    * All code coverage metrics can be gamed by using assertion free tests: aiming at a specific coverage value can increase this risk.
* What makes a successful test suite?
    * Integrated into the development cycle
        * Ideally they can be executed on the smallest change.
    * Targets only most important parts of code base
        * Most important parts to test are the business logic - the Domain Model
    * Provides maximum value with minimum costs
* Skills required to achieve goal of unit testing:
    * Understand the difference between a good and bad test
    * Be able to refactor test to make it more valuable.

## 2. What is a unit test?

* Lots of nuances in unit test practices have lead to two views on unit tests: classical and London.
    * Canonical book on classical style: Test-Driven Development: By Example
    * London style also referred to as mockist. Canonical book: Growing Object- Oriented Software, Guided by Tests
* All definitions of unit tests share these characteristics:
    * Checks a small bit of code (aka a unit)
    * Is quick.
    * Does it in an isolated manner
* The main point of disagreement between the 2 styles is how to define "isolated".
* Test double: an object with the same interface as another with a simplier implemention to facilitate testing.
    * Mocks are a subset of test doubles as are stubs
* London take on isolation: isolate a system under test from collaborators. In other words: replace all dependancies with test double.
        * Introduced in 2007 in book: xUnit Test Patterns: Refactoring Test Code
    * Advantages:
        * if a test fails, you know for sure which part of the system is failing.
        * you can have a simple rule of only testing one class at a time.
* Classical approach to isolation: unit tests should be run in isolation from each other and therefore can be run in parallel.
    * They shouldn't make use of any shared state (out-of-process dependancies): database, file system etc.
        * Use of mocks only to deal with shared dependancies.
* Summary of schools of thought:
    * ![[Pasted image 20210327110504.png]]
* Shared dependency is something that is shared between tests and gives one test the ability to affect another test.
    * Database or dependancy with mutable field.
* Out-of-process dependancy is something that runs outside execution process and provides a proxy to data not in memory: database etc
    * If you created a unique database for each test run, you could have an out-of-process dep that wasn't shared.
* Tip: tests shouldn't be verifying units of code but rather units of functionality, especially functionality that's meaningful to problem domain and can be understood by a business person.
* Classic testing may require setting up complicated dependancy graph, but in doing so, you may determine ares with code design problems.
* Classical testing may also make it more difficult to pinpoint a bug as multiple parts of the codebase can be used in one test. However, if you are running test regularly it may not be a problem.
* London style also helps guides the design process of software - you specify all the dependancies you expect to use and can slowly introduce them.
    * The downside is, the tests can be coupled to the implementation, therefore requiring you to refactor the tests if you change the implementation.
* Integration tests in 2 schools:
    * Londoner considers anything an integration test if it uses a real collaborator object
    * Classical considers anything that doesn't met the unit test criteria integration test: if it can't be run in parallel because it mutates shared deps or if it can't be run quickly.
    * Integration tests can verify multiple units of behaviour, if needed for performance reasons.
* End-to-end tests are considered subsets of integration tests.
    * UI tests, GUI tests and functional tests are usually synonyms for e2e tests.
