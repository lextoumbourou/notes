---
title: 📘 Unit Testing Principles, Practices, and Patterns - Vladimir Khorikov
date: 2021-05-24 00:00
type: book
status: draft
---

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

  ![London vs Classical Test Style](../_media/london-vs-classical-test-style.png)

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

## 3. The anatomy of a unit test

* How to structure a unit test
  * AAA pattern:
    * Split tests into 3 parts: arrange, act and assert:

        ```python
        class CalculatorTest:
            def test_sum_of_two_numbers():
                // Arrange
                first = 10
                second = 20
                calc = Calculator()
                
                // Act
                result = calc.sum(first, second)
                
                // Assert
                asert result == 30
        ```

    * Arrange section: bring the system under test (SUT) to a desired state
      * Usually largest
    * Act section: call methods on the SUT, pass prepared dependencies capture output value.
      * Usually just a single line of code.
      * If more than one line of code is required, this may indicate an "invariant violation"" an invalid state which the program can get into based on poor encapsulation of the business domain.
    * Assert section: verify the outcome.
      * Sometimes need more than one assertions, but too many assertions may indicate a missing abstraction: perhaps you need to define equality between 2 objects and compare another object.
    * This is similar to the Given-When-Then pattern, which may be more suitable for non-programmers.
    * Want to avoid multiple act and assert sections in unit tests: leave them for integration tests.
    * Avoid branching in tests: no if statements.
      * Usually a result of trying to test too many things.
      * if statements make tests harder to read.
    * Another teardown phase sometimes exists to clean up files etc. Though it's normally a method shared across multiple tests and actually not often required for unit tests.
  * Always name system under test: `sut`, so it's always clear what you're testing.
* Exploring the xUnit framework:
  * .NET framework based on jUnit (UnitTests in Python etc)
  * Tests don't have setup / teardown methods, but simply utilise a objects constructor before calling the method.
* Reusing fixtures before tests:
  * Since fixture arrangements can take up a lot space, it makes sense to reuse them but some ways of doing this are better than others.
  * Test fixture refers to an object the test runs against. It needs to stay in a consistent state, hence name: "fixture"
  * By adding fixtures to the constructor it has 2 downsides:
    * you have suddenly created tests that are coupled to each other
    * you have tests that are less readable: they don't tell the whole picture.
  * Instead, utilise private factory methods that return fixture data:

    ```c#
    [Fact]
    public void Purchase_fails_when_not_enough_inventory()
    {
        Store store = CreateStoreWithInventory(Product.Shampoo, 10); Customer sut = CreateCustomer();
        bool success = sut.Purchase(store, Product.Shampoo, 15);
        Assert.False(success);
        Assert.Equal(10, store.GetInventory(Product.Shampoo));
    }
    private Store CreateStoreWithInventory(
        Product product, int quantity)
    {
        Store store = new Store(); store.AddInventory(product, quantity); return store;
    }
    ```

    * The one exception to the rule is integration tests: can reuse some state if it's used by all tests, like a database connection.
* Naming unit tests:
  * Prominent but not helpful style:

    ```
    [MethodUnderTest]_[Scenario]_[ExpectedResult]
    ```

  * This style is unhelpful because you focus on implemention details, not functionality.
  * Guidelines:
    * Use simple English phrases.
    * Don't follow strict naming policy: allow freedom of expresion.
    * Name test as if describing scenario to non-programmer
    * Seperate words with underscores.
    * No need to use name of SUT method in test name, unless testing utility code.
    * Using `should` in the test name is another anti pattern: a test is a statement of fact. Use `is`
    * Use proper grammar in test names.
* Refactoring to parameterized tests
  * Some tests that test related functionality can be grouped. The functionality that does that is called: parameterized tests
  * Example from nose in Python:

     ```python
     class Test(unittest.TestCase):
 
         @params((1, 2), (2, 3), (4, 5))
         def test_less_than(self, a, b):
             assert a < b
     ```

  * Parameterisation does impact readability.
  * Rule of thumb:
    * Only keep positive and negative cases together if it's clear from input params what cases stand for
    * If it's too complicated, split into separate methods.
* Assert library can be used for even more readability.

## 4. The four pillars of a good unit test

* Good unit tests has the following attributes:
  * Protects against regressions
    * The more code you add, the more chance that existing code will stop working as intended: "code isn't an asset, it's a liability"
    * 2 main features of code helps this attribute:
      * Amount of code executed in each test.
      * Complexity of code tested.
      * Code's domain significance.
    * Trivial code is rarely worth testing
    * Libraries and frameworks should be tested as part of your suite to verify your assumptions about their behaviour.
  * Resistance to refactoring
    * Tests are the main tool to provide resistance to refactoring with:
      * early warnings you've broke functionality.
      * confidence code won't lead to regressions
    * False positive: when tests raise error but no feature is broken
      * Degrades confidences in test suite and causes you to ignore positives
      * Fewers of these the better: less false positives = more resistance to refactoring
    * Test should not be coupled to implementation details, it should verify end results.
      * The more coupled to implementation, the less resistant to refactoring.
      * Uncoupled tests may still fail when you refactor, but the breakage should be picked up by the compile, not the tests.
  * Fast feedback
    * When tests fail quickly, you dramatically reduce feedback loop
    * Slow tests can be run less often.
  * Maintainability
    * How hard is it to read tests?
    * How hard is it to actually run tests - ie is setup fully automated etc.
* To determine impact of individual test, rate it from 0 to 1 for each metric and multiple for final score.
* Some of the metrics are mutually exclusive, so cannot expect to get perfect score: as you increase resistance to refactoring, you decrease fast feedback: ie using a database mock will increase the test speed but decrease resistance to refactoring.
* Writing the ideal test:
  * The first 3 attributes of a good test: protects against regressions, resistant to refactoring and fast feedback are mutually exclusive.
  * Maintainability is usually not mutually exclusive so should be maximised, though e2e tests usually require more code and thus score lower by default.
  * Since all attributes cannot be maximised, write tests that result in no metric at 0.
  * Resistance to refactoring is least negotiable because it either is or isn't, so trade off will tend to be between protection against regression or fast feedback.
* Test Pyramid advocates for a ratio of tests where unit tests is greated, integration in middle and least e2e. It is also ordered so closeness to customer is at tip of pyramid, where e2e tests are.
* Exceptions to Test Pyramid:
  * CRUD applications that mainly interact with DB are usually best served with mostly integration tests.
  * Other domains with low algorithmic or business complexity also requires less unit tests.
  * APIs that exposure interactions with DB could be served entirely with e2e tests or a hybrid of e2e and integration.
* Black box vs white box:
  * Black box tests tests functionality with no knowledge of internal structure
  * White box the opposite: test with knowledge of internals.
  * Since aim is to maximise resistance to refactoring, pick black box by default. Use white box for test analysis: ie branch coverage metrics, and then write black box armed with that knowledge.

## 5. Mocks and test fragility

* Test double is overarching term for any non-prod fake dependacies in tests.
* Gerard Meszaros in xUnit Test Patterns, claims there are 5 variations of test doubles:
  * dummy
  * stub
  * spy
  * mock
  * fake
* In reality, there are only 2 types: mock **and** stub. Each of the 5 variations are really just variants of the 2.
  * Mocks are used for checking *outgoing interactions*
  * Stubs are used for *incoming interactions*

 ```mermaid
 graph TD;  
  Double-->Mock("Mock: (mock, spy)");  
  Double-->Stub("Stub (stub, dummy, fake)");
 ```

  * Spies are the same as mocks but written manually. Also known as: handwritten mocks.
  * Dummy is a simple hardcoded value, like a string or None used to satisfy an interface.
  * Stub is a more sophisticated.
  * Fake is a stub for a dependancy that doesn't exist.
* Rule of thumb: you don't assert interactions with stubs as it leads to test fragility.
  * Another term for verifying implementation details instead of end result = overspecification
* Sometimes same double will be mock and stub. If different methods are used, then haven't violated the rule.
* Mocks and stubs are related to the Command Query Separation (CQS) Pattern.
  * Command are methods with side effects - generally they should not return a value (some exceptions to this like `array.pop()` methods that return the popped item)
  * Query is the opposite - it has no side effects and is used to return a value to client.
* Observable behaviour vs implementation details
  * All prod code be be categorised along 2 dimensions:
    * Public or private API
    * Observable behaviour vs implementation details
  * Private code is usually method or attribute declared with private keyword (if applicable to language)
  * Observable more nuanced. Traits are either:
    * Exposes an operation (calculation or side effect) that lets client do something.
    * Exposes a state (condition of a system) that lets client do something
  * If neither traits, then it's an implementation detail
* When observable behaviours coincides with public API and all implementation details are private, we consider it a [Well-Designed API](../../permanent/well-designed-api.md)
  * Simple way to tell if class leaks implementation detail, if the number of operations client has to invoke to achieve goal is greater than 1.
  * A well designed API automatically improves unit tests.
* [Encapsulation](../../permanent/encapsulation.md)
  * Act of protecting against invariant violations
    * By leaking implementation details you run the risk of exposing invariant violations (ie allowing data to get into incorrect state)
  * Good encapsulation protects against errors as the complexity of the code base increases: "You cannot trust yourself to do the right thing all the time - so, eliminate the very possibility of doing the wrong thing"
    * Related to Martin Fowler's [Tell Don't Ask](https://www.martinfowler.com/bliki/TellDontAsk.html#:~:text=Tell%2DDon't%2DAsk,an%20object%20what%20to%20do) principal.
* Relationship between mocks and test fragility
  * Hexagonal architecture (term introduced by Alistair Cockburn)
    * Applications consist of 2 layers: domain and application services
      * Domain layers == business logic.
      * Application is for orchestrating communication to domain layer.
    * Business logic is most important part of application and shouldn't mix with application.
    * Tests can have fractal structure: verify behavior that helps achieve same goals at different levels.
  * Intra-system vs inter-system communication
    * Intra-system == communication between classes within your app
    * Inter-system == communication between other applications
  * Intra-system are part of the implementation detail and shouldn't be mocked.
  * Inter-system can sometimes be part of observable application behaviour: for example, sending email after adding user, adding a message to queue
    * Use of mocks here is a good idea, to ensure contract is maintained.
* Not all out-of-process dependancies should be mocked
  * Although tests should be optimised for running in parallel, some out-of-process deps like databases are entirely implementation details and not important to the client achieveing their goals. They shouldn't be mocked (more on how to do this in upcoming chapter.)

## 6. Styles of Unit Testing

[@khorikovUnitTestingPrinciples2020]

[//begin]: # "Autogenerated link references for markdown compatibility"
[Well-Designed API]: ../../permanent/Well-Designed API "Well-Designed API"
[Encapsulation]: ../../permanent/Encapsulation "Encapsulation"
[//end]: # "Autogenerated link references"
