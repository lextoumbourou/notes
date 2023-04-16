---
title: Arrange, Act and Assert (AAA) Pattern
date: 2021-03-28 00:00
tags:
  - AutomatedTesting
summary: A pattern for structuring unit tests
---

A pattern for structuring unit tests, where tests are broken up into 3 sections:

* Arrange: where you prepare any fixtures and setup the test.
* Act: usually a single method call which calls the system under test.
* Assert: the final assertions which are run after act.

The system under test is always stored in a variable called `sut` .

Khorikov writes, by conforming to a standard like this, anyone can easily read and understand a test reducing maintenance cost.

[@khorikovUnitTestingPrinciples2020] *(pg. 42-43)*

Example:

```python
def test_datestring_is_formatted():
    # Arrange
    date_time_instance = datetime.datetime(2008, 1, 2)
    sut = DateFormat(date_time_instance)
    
    # Act
    result = sut.format()
    
    # Assert
    assert result == "2008-1-2"
```

Similar to the [Given-When-Then](https://martinfowler.com/bliki/GivenWhenThen.html) pattern.
