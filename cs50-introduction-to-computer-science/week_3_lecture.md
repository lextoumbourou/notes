Monday
======

* Terany operator, takes 3 values. Opposed to binary or unerary.
* No notion of package names or namespace in C. Collisions happen.
* In C, main function gets put in memory first.
* An array in C is a bunch of contiguous memory in C.
* Core dump is a file with contents of memory dumped when a fault occurs
* Good practise in C, is to use 
```#define GLOBAL val``` 
to define global variables
* C modules go in module.h files

Wednesday
========

* Linear approach to search may be to go through first, then second, then third etc
* Logarithmic would generally be better
* Linear relationship is generally 1 to 1. As size of problem grows, so does time to complete.
* n generally refers to size of problem
* Notation uses O() to define algorithm efficiency.
* O defines worst case.
* Omega best case
* O(n)
Example: David finding tallest person 1 at a time. 1 to 1. Linear.
* O(n/2)
Example: David counting in pairs. 1 to 2. Still linear, just faster.
* O(log2 n)
Example: People work out if they're taller than person next to them. If so, stand. Else, sit. As problem set grows, time to complete does not. This is how you should be aiming to build algorithms.

* Any algorithm that uses constants like n*2 or n/2, we tend to just remove the /2 or *2 because computer speeds are always improving and stuff

## Paper board example

* You could randomly attack the problem, but then you have to keep track where you've been
* To make it easier, you could arrange in some sort of order, then attack the problem with that assumption
* If it was sorted, you could take a number from the middle to get greatest range (divide in half). Then you can remove one half, then get the next. If you doubled the problem, you would only need to add one more division.

## Sort examples

### Selection sort 0(n**2)
    * expensive. 
    * Find the lowest, move it to the start, swap 0th value with it and so on.
    * Could work n+n-1+n-2+n-3 etc
    * Best case: Omega(n**2)

### Bubble sort 0(n**2)
    * Less expensive in some instances
    * Worst possible scenario is reverse order = n**2
    * Named Bubble because they "Bubble up" toward the end
    * Keep a counter of swaps, if no swaps, then they're sorted
    * Best case: Omega(n)

### Merge sort 
    * Appears far superiour
