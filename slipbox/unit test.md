The definition of a unit test, according to Khorikov, is an automated test which meets the following criteria:

* Verifies a single unit of behaviour
* Executes quickly
* Runs in isolation from other tests.

Two schools of thought on unit testing: [[London (mockist)]] and [[Classical (Detroit)]], differ mainly on the intepretation of the third point. The former views isolation as being from all dependancies: use of mocks should ensure only one class is tested at a time whereas the classic refers to isolation between tests and their ability to parallelise.

On the other hand, an [[integration test]] is simply any test which can't satisfy this criteria.

---

Tags: #AutomatedTests #UnitTests 
Reference: [[Unit Testing Principles, Practices and Patterns#2 What is a unit test]]