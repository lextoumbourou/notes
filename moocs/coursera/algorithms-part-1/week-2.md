# Stacks

* Stack

  * Take out item most recently used (LIFO).
  * ``push`` and ``pop``.

* Queue

  * Take out least recently used (FIFO).
  * ``enqueue`` and ``dequeue``.

## Stack API Implementations

* Linked list implementation:

   * [LinkedStackOfStrings.class](./code/java/LinkedStackOfStrings.class)
   * Preposition: every operation takes constant time in the worst case.
   * Proposition: a stack with ``N`` items uses 40 ``N`` bytes.
     * Object overhead = 16 bytes.
     * Inner class extra overhead = 8 bytes.
     * Reference to String = 8 bytes.
     * Reference to Node = 8 bytes.

   <img src="./images/linked-list-stack-memory-use"></img>

* Array implementation:

  * [FixedCapacityStackOfStrings](./code/java/FixedCapacityStackOfStrings.class)
  * Fundamental defect: declare size of array ahead of time.
  * Extra considerations:
    * Underflow: throw exception if pop from empty stack.
    * Overflow: throw exception if push to full stack or resize.
    * Loitering: holding reference to object when no longer needed.
