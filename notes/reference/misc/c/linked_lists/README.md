# Linked Lists

Questions:

  1. What are linked lists?
  2. What are advantages to using Linked Lists over arrays?
  2. What are some of the disadvantages?

Answers:

  1. Dynamically growing arrays.
  2. They can grow dynamically, an item can be added to the middle.
  3. Disadvantages:
      1. No random access: can't reach the nth item in the array.
      2. Dynamic memory allocation: risk of memory leaks and segment faults.
      3. Larger memory overhead than arrays.

## Explained in English

* Dynamically allocated "nodes" arranged such that each node contains one value and one pointer.
* Pointer always points to next member of the list.
* If pointer is null, it's the last element in the list.

Node definition:

```
typedef struct node {
    int val;
    struct node * next;
} node_t;
```

## Code Example

[example.c](example.c)
