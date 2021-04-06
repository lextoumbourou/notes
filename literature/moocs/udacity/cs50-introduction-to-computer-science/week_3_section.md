* Get problem Set 3 [Walkthrough](https://www.cs50.net/psets)
* Don't create unneccessary variables (except for readability)
* In Problem Set 1, ensure you're checking the input

## Design
* Reason in psuedo code (or tests)
* Break code up into functions

## Arrays (in C)
* A set of elements of the same type
* Contiguous in memory
* Each elem is accessed with index value
* Passed by reference (not by value) to functions. Just need its name.
* 0-indexing is used in array because it's how far away in memory you are from the first item. So, first item is *1st location*+0, 2nd is *2nd location*+1 and so on.

```./test best mest lest```
* argc == 4
* argv[0] == "test"
* argv[1][2] == "s"
* argv[3][4] == IndexError (in Python), null terminator (in C)

## Running time
* How long does it take an algorithm to run
* Not in terms of time, in terms of steps
* One algorithm may solve problem faster than other, as size of problem increases, it may be way faster
* Asymptotic notation allows us to represent and compare these running times

## Asymptotic Notation
### "Big O"
* "Worst case running time (upper bound)
* "Most important when classifying speed of algorithm

### "Omega"
* Best case running time (lower bound)

### "Theta"
* Average case running time (upper and lower bound combined)

## Comparisons
In order of best to worst:
1. O(1) - constant
2. O(log n) - logarithmic
3. O(n) - linear
4. O(n**2) - quadratic
5. O(n**c) - polynomial
6. O(c**n) - exponential
7. O(n!) - factorial

Examples of bad time:
```for i in len(my_str)```
* Runs len everytime it iterates, causing it to grow exponentially every time the iteration runs.

## Binary Search
* List needs to be sorted
* Logarithmic time
* To search a sorted list:
1. Start in the middle. 
2. Compare two numbers. 
3. Remove one half.
4. Repeat

## Bubble Sort
1. Compare your neighbour, if smaller, swap.

## Recursion
* Factorial is all previous numbers multiplied, so factorial(5) = 5*4*3*2*1





