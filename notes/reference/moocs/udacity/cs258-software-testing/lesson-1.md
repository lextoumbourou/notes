---
title: "Lesson 1: What Is Testing?"
date: 2014-12-14 00:00
category: reference/moocs
parent: uol-discrete-mathematics
status: draft
---

# Lesson 1: What Is Testing?

* Introduction
  * Finding failures in software and fixing them will allow us to eventually run out of bugs
  * Question: How can we get rid of all bugs in software if Google and Microsoft can't?
  * Answer: Breaking problems of testing down into smaller problems
* What is testing?
  * Test inputs
  * Software
  * Output
    * is it what you want?
* What happens when we test software?
  * Run experiment can have two possiblities
    * Output okay -- we haven't learned a lot
    * Output bad -- we learn about a lot and have found a bug in our software we can fix
      * Bug in S.U.T. (software under test) -- easy!
      * Bug in Acceptability Test
      * Bugs in specification
      * Bugs in OS compiler, libraries, hardware -- much harder and more painful to work around
      * Bugs in ??
* Mars Climate Orbiter bug
  * Unit error caused Orbiter to crash
    * Metric expected == metres per second
    * Actual metric received == feet per second
  * Probably a bug in the specification
* Fixed Sized Queue
  * enqueue
  * dequeue
  * FIFO order
  * enqueue(7); enqueue(8)
  * dequeue gets 7 then 8
* Equivalent tests
  * Basic idea: Single test maps single input to an output
  * Big idea: need to find ways to ensure input represents a mapping of as big an output as possible.
    * Input should represent a "class of inputs"
      * If code executes correct for the one input, it should work for all in the class.
* Testing the Queue
  * ```test2()``` -- should not be equivalent to ```test1()```
  * ```test3()``` -- should not be equivalent to ```test1()``` or ```test2()```
* Creating Testable Software
  * Clean code
  * Refactor
  * Should always be able to describe what a module does and how it interacts with other code
  * No extra threads
  * No swamp of global variables
  * No pointer "soup"
  * Modules should have unit tests
  * Where applicable, support fault injection
  * Assertions!
* Assertions
  * Check for a property that must be true

  ```
  def sqrt(arg):
      ... compute result
      assert result >= 0
      return result
  ```

  * Rule 1: Assertions are not for error handling
  * Rule 2: No side effects!

  ```
  assert change_some_global_var() == 0
  ```

    * Optimisations in Python will drop all assertions
  * Rule 3: No silly assertions

  ```
  assert (1 + 1) == 2
  ```

* Check rep, the exercise
  * Add additional assertion to ```checkRep()``` to ensure:
    * Catches bug in ```enqueue()``` before it can misbehave
    * Violates relative check between head, tail and size.
* Why assertions
  * Make code self-checking leading to more effective testing
  * Make code fail early, closer to the bug
  * Assign blame
  * Document assumptions, preconditions, postconditions, invariants
* Assertions in production
  * GCC == ~900 assertions
  * LLVM == ~13000 assertions
  * 1 assertion per 110 L.D.C.
* Disabling assertions in production code?
  * Pros
    * Code runs faster
    * Less likely to stop running (really depends on problem domain)
  * Cons
    * If assertions cause side effects, shit could break!
* Domains and Ranges
  * Domain == set of possible inputs
    * We should tests programs with values sampled from their domain
  * Range == set of possible outputs
* Good test cases
  * "Interfaces that span trust boundaries are apecial and must be tested on the full range of representable values"
* Testing a GUI
  * Domain == set of all possible GUI actinos
  * Range == set of possible GUI application states
* Timing dependant problems
  * Software cares about the time inputs arrive
  * Example: double click different to two spaced clicks
  * Example: [Therac 25](http://en.wikipedia.org/wiki/Therac-25)
    * Race condition:
      * As operators became better with machine, they got faster with tests and started triggering a race condition bug
* Non-functional inputs
  * Context switches
    * Switches between different threads of execution
* Kinds of testing (a survey)
  * White box
    * Tester knows a lot about system internals
  * Black box
    * Opposite
  * Unit testing
    * Testing only a small amount of software at a time
    * Usually white box
    * Mock objects - ya know what this is :)
  * Integration testing
    * Testing multiple software modules, already tested, together
  * System testing
    * Does system as a whole work?
      * Usually black box testing
  * Differential testing
    * Take same test input and deliver to two different implementations of SUT
  * Stress testing
    * Test system at or above its normal usage level
  * Random testing
    * Feeding random data into software
