Monday
=====

## Merge sort
```On input of n elements
    If n < 2
        Return
    Else
        Sort left half of elements
        Sort right half of elements
        Merge sorted halves```

* Calls itself asking to split list in half until problem is down to just one item to sort
* Recursion: answer a question with the same question.
* Requires an empty array to merge items into; doubles memory requirements
* Running time 
* Base case:
T = 0, if n < 2
* Recursive case:
T(n) = T(n/2) + T(n/2) + n, if n > 1
With numbers:
T(16) = 2*T(8) + 16
T(8) = 2*T(4) + 8
T(4) = 2*T(2) + 4
T(2) = 2*T(1) + 2
T(1) = 0

## Sigma
* Function that adds previous numbers

Wednesday
========

* Hexidecimal numbers all begin with 0x
```0x01```
* NULL is a pointer to memory address 0 in C
* Malloc is a function to allocate memory, returns first byte of memory
* Pointers are setup initial without pointees
* Dereferencing points a pointer to a pointee
