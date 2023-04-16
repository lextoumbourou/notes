---
title: "Lesson 3: Random Testing"
date: 2014-12-14 00:00
category: reference/moocs
parent: uol-discrete-mathematics
status: draft
---

# Lesson 3: Random Testing

* Random testing
  * Takes pseudo-random numbers as input
  * Start with seed to ensure results are consistent
  * Need a lot of "domain knowledge"
  * Wrap in driver script that runs it automatically while you sleep :)
* Testing Compilers
  * Random test generators frequently cause program timeouts, so you need to ensure that program is killed after a timeout
* Input validity problem
  * Thinking about where you want your random code to fail
  * Browser example
    * Random bits == mainly failures around protocol handling (ie HTTP headers invalid)
    * HTTP protocol correct and HTML tokens == mainly fails in HTML parsing
    * Etc
* Generating valid credit card numbers
  * Randomly generate valid credit card numbers as input domain to stress test validity function
  * Credit card numbers:
    * First 6 - issuer identifier
    * Last digits - compared against "check digit" as per Luhn's algorithm
* Luhn's Algorithm
  * Card with even number of digits:
    * double every odd numbered digit
    * If product is more than 9, subtract nine
  * If card has odd number of digits, double even digits
* Luhn's Algorithm exercise
  * Generate takes two params
    * ```prefix``` - issuer identifier
    * ```length``` -
* Problems with Random Tests
  * Think about which areas you want to be stressing
  * Stressing programs input validity testing unnecessarily
    * Need to be conscious of generating too many bug tickets
    * Some programs won't have good validity checks due to perform and other reasons
    * Need to think hard about good random test generators
* Books to check out:
  * Testing for Zero Bugs, Ariel Faigon
  * Random Testing, by R. Hamlet
* Random Testing vs Fuzzing
  * Fuzzing originally referred to random testing
    * Original fuzzing paper (1990)
      * They found random testing allowed them to crash 1/4 to 1/3 of utilities without doing any input validity checking.
    * Fuzz revisited (1995)
      * Extended to:
        * Network apps
        * GUI apps
      * Similar results
    * Windows fuzzing (2000)
      * Many apps can be crashed
    * MacOSX fuzzing (2006)
      * Low crash rate for cmd line apps
      * 22/30 GUI apps crashed
  * From the early 2000s onward, fuzzing referred to network penetration testing through similar techniques
* The Queue (Quiz)
  * Start off by creating a queue
  * After each queue operation, random test should call ``checkRep()```
  * Ensure that enqueues fails when queue is full and succeeds when not
  * Ensure that dequeue fails when queue is empty and succeed when not
  * Keep track of values that come out of dequeue operations
* Generating Random Input
  * "Generative random testing"
    * "inputs are created from scratch"
  * "Mutation-based random testing"
    * "inputs are created by modifying non-randomly generated test cases"
* Mutation-based random tester
  * Start with known input
  * Randomly modify it but keep it within the domain of original input
  * Examples (I can think of)
    * Push JSON to API with fields randomly changed with random unicode chars
    * Update database with random chars for each field
    * Save random blobs on filesystem
* [Charlie Miller's "Babysitting an Army of Monkeys"](https://www.youtube.com/watch?v=Xnwodi2CBws)
* Oracles for Random Testing
  * If oracle isn't automated, you don't have an oracle
  * Weak oracles
    * Crashes
    * Violation of language rules (attempt to access list index that doesn't exist)
  * Medium Oracle
    * Assertions
      * Provides good application specific checking
      * Doesn't guarantee correct operation
  * Strong Oracle
    * Different implementation of same spec
    * Differential testing of compilers
    * Older version of software we're testing
    * Reference implementation
      * Ie new Python compiler reference against C Python compiler
    * Function inverse pair
      * Compress / decompress
      * Save / load
      * Transmit / receive
      * Encode / decode
    * Null space transformation
      * change program in such a way that the output of the prog should not change and run through same interpreter

# Problem Set 3 - Sudoku Tester

* [Sudoku checker](http://forums.udacity.com/answer_link/100237967/#cs258)
