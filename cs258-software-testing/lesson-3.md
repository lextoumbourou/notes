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
* Random Testing Alternative Histories
