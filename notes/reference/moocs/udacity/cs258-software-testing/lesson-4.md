---
title: "Lesson 4: Random Testing in Practice"
date: 2014-12-14 00:00
category: reference/moocs
parent: uol-discrete-mathematics
status: draft
---

# Lesson 4: Random Testing in Practice

* Random Testing in the Bigger Picture
  * Why does random testing work?
    * Based on weak bug hypothesis
    * People make same mistakes coding which testing
    * Huge asymmetry between speed of computers and people
  * Why random testing is effective on commercial software?
    * Because developers aren't doing it enough
* Tuning rules and probabilities
  * Start simple -> examine test cases, think hard
* Filesystem testing
  * Special-case mount & unmount
  * Keep track of open files
  * Limit size of files
* Fuzzing the bounded quque
  * Looking at queue as finite state machine
  * Empty -> 1 -> 2 -> fill
* Fuzzing Implicit Inputs
  * "Non-API inputs to SUT that affect its behaviour"
    * Doing stuff like: generating load, generating network activity to potentially affect your application.
  * "Unfriendly emulators"
    * Stuff that removes line from cache
    * Does other stuff to break app
* Can Random Testing Inspire Confidence?
  * If you have:
    * Well-understood API +
    * small code +
    * strong assertions +
    * mature, tuned random tester +
    * good coverage results -
    * confidence
  * Otherwise, you should also use other testing methods.
* Tradeoffs in spending time on random testing
  * Cons
    * Input validity can be hard
    * Oracles are hard too
    * No stopping criterion
    * May find unimportant bugs
    * May find same bugs many times
    * Can be hard to debug when test case is large and/or makes no sense
    * Every fuzzer finds different bugs
  * Pros
    * Less tester bias, weaker hypotheses about where bugs are
    * Once testing is automated, human cost of testing goes to zero
    * Can tell us stuff that surprises us about our software
    * Every fuzzer finds different bugs

# Problem Set 4: Fuzzer

* Adding random bytes to an MP3 to try to break ``afplayer``: [fuzzer.py](fuzzer.py)
