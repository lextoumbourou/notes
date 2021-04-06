To do: rewrite.

Notes from (You are naming your tests wrong!)[https://enterprisecraftsmanship.com/posts/you-naming-tests-wrong/]


* Prominent but not helpful test convention:

```
[MethodUnderTest]_[Scenario]_[ExpectedResult]
```

Where:

* MethodUnderTest is the name of the method you are testing.
* Scenario is the condition under which you test the method.
* ExpectedResult is what you expect the method under test to do in the current scenario.

* With simple phrases, you can describe the system behavior in a way that’s meaningful to a customer or a domain expert.

* Cryptic names impose a cognitive tax on everyone, programmer or not.

---

Adhere to the following guidelines to write expressive, easily readable test names:

No rigid naming policy. You simply can’t fit a high-level description of a complex behavior into a narrow box of such a policy. Allow freedom of expression.

Name the test as if you were describing the scenario to a non-programmer who is familiar with the problem domain. A domain expert or a business analyst are good examples.

Separate words by underscores. It helps improve readability, especially of long names.


Also notice that although I use the pattern [ClassName]Tests when naming test classes, it doesn’t mean that the tests are limited to verifying only that ClassName. The unit in unit testing is a unit of behavior, not a class. This unit can span across one or several classes, the actual size is irrelevant. Still, you have to start somewhere. View the class in [ClassName]Tests as just that: an entry point, an API, using which you can verify a unit of behavior.


Remember, you don’t test code, you test application behavior.

The only exception to this guideline is when you work on utility code. Such code doesn’t contain business logic - its behavior doesn’t go much beyond simple auxiliary functionality and thus doesn’t mean anything to the business people. It’s fine to use the SUT’s method names there.


Improving test name:

```
[Fact]
public void IsDeliveryValid_InvalidDate_ReturnsFalse()
{
    DeliveryService sut = new DeliveryService();
    DateTime pastDate = DateTime.Now.AddDays(-1);
    Delivery delivery = new Delivery
    {
        Date = pastDate
    };
    
    bool isValid = sut.IsDeliveryValid(delivery);
    
    Assert.False(isValid);
}
```

First attempt to write it out into plain English:

```
public void Delivery_with_invalid_date_should_be_considered_invalid()
```

"Don’t include the name of the SUT’s method into the test’s name. Remember, you don’t test code, you test application behavior."


 be specific and reflect this knowledge in the test’s name:

public void Delivery_with_past_date_should_be_considered_invalid()


The wording should be is another common anti-pattern. A test is a single, atomic fact about a unit of behavior. There’s no place for a wish or a desire when stating a fact. Name the test accordingly. Replace should be with is:

```
public void Delivery_with_past_date_is_invalid()
```

And finally, there’s no need to avoid basic English grammar. Articles help the test read flawlessly. Add the article a to the test’s name:



Don’t use a rigid test naming policy.

Name the tests as if you were describing the scenario in it to a non-programmer familiar with the problem domain.

Separate words in the test name by underscores.

Don’t include the name of the method under test in the test name.