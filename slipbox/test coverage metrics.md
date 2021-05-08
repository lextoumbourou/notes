Title: Test Coverage Metrics
Date: 2021-03-24
Tags: #AutomatedTesting

---

Test coverage metrics are a tool used to define how much of a code base is covered by a unit test suite. 2 main metrics used:

  * Code coverage:
  
     $$\text{Code coverage (text coverage)}=\frac{\text{Lines of code executed}}{\text{Total number of lines}}$$
     
     Khorikov says that the major downside is that code can simply be shuffled to trick the metric into increasing without increasing test coverage.

 * Branch coverage:
 
   $$\text{Branch coverage}=\frac{\text{Branches traversed}}{\text{Total number of branches}}$$
   
   Improves upon code coverage, as branches will still be examined whether one line or multiple. The downside, according to Khorikov, which covers all code coverage metrics is that tests without assertions will be counted as branch traversal.
   
---
   
References:
* [[Unit Testing Principles, Practices and Patterns|# Unit Testing Principles, Practices, and Patterns]]