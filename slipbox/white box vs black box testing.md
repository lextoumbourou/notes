Title: White Box vs Black Box testing
Date: 2021-04-04
Tags: #AutomatedTesting

---

Blackbox tests are tests that run without knowing anything about the internal structure of the software. They test what the software is expected to do from a business perspective.

Whitebox tests are the opposite of that. They test with a good understanding of the internal workings of the software.

Khorikov argues that since black box tests provide the best resistance to refactoring one of the [[4 Pillars of Good Unit Tests]], you should aim to mostly write tests in this style. Reserving white box testing for the analysis of your tests, for example, by utilising [[Test Coverage Metrics]] to examine branch or code coverage.

---

References:
* [[Unit Testing Principles, Practices and Patterns#4 The four pillars of a good unit test]]