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