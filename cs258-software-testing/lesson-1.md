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
