## Memory

* Code/data stored in random-access memory (RAM)
* Hexadecimal numbers - base 16 (0 through 9 then A through F)
* Every set of 4 bits can represent 1 hex digit
* To signal we're using hexadecimal, we start wiith 0x
Number examples:
0x7 = 7
0xE = 14

## Stack
* A part of memory
* Need to store something? Put it on top.
* Done with it? Take it off.
* Each function called gets own block, called "Frame"
* Local funntion variables go in the frame
* When function returns, frame becomes inaccessible.

## Pointers
* Data stored in mem has value and address
* Pointer is a variable that stores a mem address
* Every mem address is 4 bytes (32 bits - 8 hexadecimal nums)
```int x = 5
int* y = &x;```
* y holds the address of x

## Heap
* Dynamic Memory Allocation: requesting memory on the fly:
```void* malloc(int <number of bytes>)```
* Reserves a block memory on the heap
* Returns the address of this block
```sizeof(<data type>)```
Returns the number of bytes a given type occupies
```void free(void* <name of pointer>)```
* Frees up the reserved memory

* An int is 4 bytes
* A char is 1 bytes
* A pointer is 4 bytes

## Stack vs Heap
###Heap
* contains global variables
* dynamically allocated memory

###Stack
* contains local variables
* function calls create new frames

* When running ```malloc``` you should check that it doesn't return NULL

## Merge Sort
* A faster sort
* Halving time is logarithmic
* Sorting time is n
Big O
* O(n log n)
* Breaking a list in half and rebuilding it = O(log n)
* Sorting each half = O(n)

