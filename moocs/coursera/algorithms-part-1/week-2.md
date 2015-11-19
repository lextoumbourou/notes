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

   * Trade offs against resizing array:
     * Every operation takes constant time in *worst case*.
     * Extra space to deal with links.
     * Use if extra space isn't an issue, but want to ensure all operations are constant time.

* Array implementation:

  * [FixedCapacityStackOfStrings.class](./code/java/FixedCapacityStackOfStrings.class)
  * Fundamental defect: declare size of array ahead of time.
  * Extra considerations:
    * Underflow: throw exception if pop from empty stack.
    * Overflow: throw exception if push to full stack or resize.
    * Loitering: holding reference to object when no longer needed.

* Resizing arrays

  * Problem: requiring client to provide capacity does not implement API
  * Question: how to grow and shrink array?

  * First try:
    * ``push()`` - increase size of array of ``s[]`` by 1.
    * ``pop()`` - decrease size of array ``s[]`` by 1.
    * Too expensive:
      * Need to copy all items to a new array.

  * Challenge:
    * Ensure that array resizing happens infrequently.
  
  * Solution:
    * "If array is full, create an array of twice the size."
    * [ResizingArrayStackOfStrings.class](./code/java/ResizingArrayStackOfStrings.class)
    * Inserting first ``N`` items takes time proportional to ``N`` (not ``N**2``).
    * "Amortized" - reduce or pay off with regular payments.
    * "Amortized analysis" - "consider the total cost averaged over all operations".
      * In other words: every operation takes constant time, 'cept for one worst case when it's ``N`` and you "pay off" the debt, as it were.
    * Calls to ``resize`` happen everytime ``N`` doubles: logarithmic.

  * Question: How to shrink array?
  
  * First try:
    * ``push()`` - double size of array s[] when array is full.
    * ``pop()`` - halve size of array s[] when array is one-half full.
    
    * Problem: Too expensive in push-pop-push-pop scenario (thrashing)
      * Will be constantly growing and shrinking the array.
    * Solution: Halve array when it's *one-quarter* full.
      * Array is between 25% and 100% full.
      * Amount of memory is always a constant multiple of items on the stack.
      * |Question| how do you avoid "thrashing" when shrinking array in resize implementation?
    * Trade offs with Linked List:
      * Every operation takes constant *amortized* time: eg it's constant time but you have to "pay off" the debt shrinking or growing the array.
      * Less wasted space.
      * Use if implementing something to conserve space, where occasional slowness of operations isn't an issue.

* |Question| Trade off between Linked List implementation or Resizing-Array implementation?

## Queues

* Queue API:

```
public class QueueOfStrings

  // Create an empty queue
  QueueOfStrings()

  // insert a string onto the queue
  void enqueue(String item)

  remove and return the last string
  String dequeue()

  is the queue empty?
  boolean isEmpty()

  number of strings on the queue
  int size()
```

* Queue: linked-list representation

  * Maintain pointer to first and last nodes in a linked list; insert / remove from opposite ends.
  * [LinkedListQueue.class](./code/java/LinkedListQueue.class)
