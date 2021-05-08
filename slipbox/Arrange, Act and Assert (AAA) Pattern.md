Title: Arrange, Act and Assert (AAA) Pattern
Date: 2021-03-28
Tags: #AutomatedTesting

---

A pattern for structuring unit tests, where tests are broken up into 3 sections:

* Arrange: where you prepare any fixtures and setup the test.
* Act: usually a single method call which calls the system under test.
* Assert: the final assertions which are run after act.

The system under test is always stored in a variable called `sut` .

Khorikov writes, by conforming to a standard like this, anyone can easily read and understand a test reducing maintenance cost.

Example:

```
def test_datestring_is_formatted():
    # Arrange
    date_time_instance = datetime.datetime(2008, 1, 2)
    sut = DateFormat(date_time_instance)
    
    # Act
    result = sut.format()
    
    # Assert
    assert result == "2008-1-2"
```

Similar to the [[Given-When-Then]] pattern.

---

References:
* [[Unit Testing Principles, Practices and Patterns|# Unit Testing Principles, Practices, and Patterns]] 