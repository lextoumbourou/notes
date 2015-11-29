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

* Linked-list representation:

  * Maintain pointer to first and last nodes in a linked list; insert / remove from opposite ends.
  * [LinkedListQueue.java](./code/java/LinkedListQueue.java)

* Resizing array implementation:

  * Use array ``q[]`` to store items in queue.
  * ``enqueue()``: add item at ``q[tail]``.
  * ``dequeue()``: remove item from ``q[head]``.
  * Update ``head`` and ``tail`` modulo the capacity.
  * [ResizingArrayQueue.java](./code/java/ResizingArrayQueue.java)

## Generics

* Avoid having multiple implementations for each type of data.
* Avoid having to cast ``Object``s everywhere.
* Client example:

```
Stack<String> s = new Stack<String>();
String a = "Hello";
String b = "World";
s.push(a);
s.push(b);
a = s.pop();
```

* Java doesn't allow generic array creation. Required to cast:

```
s = new Item[capacity]; // Invalid!
s = (Item[]) new Object[capacity]; // Fixed
```

* Primitive types all have a *wrapper* object in Java. The convention is to use the capitalize version like ``Integer``.

## Iterators

* ``Iterable``: class that has a method that returns an ``iterator()``.
* ``Iterator``: class that has methods ``hasNext()`` and ``next()``.
* Can use the foreach shorthand:

```
for (String s : stack)
    StdOut.println(s);
```

equivalent to:

```
Iterator<String> i = stack.iterator();
while (i.hasNext())
{
    String s = i.next();
    StdOut.println(s);
}
```

# Elementary Sorts

## Sorting Introduction

* Problem summary: rearrange array of N items into ascending order.
* Goal for clients: one client to sort *any* type of data.
  * Requires a *Callback* (aka a reference to executable code).
  * Java uses *interfaces* for implementing callbacks.
* Object implementation:

  ```
  public interface Comparable<Item>
  {
      public int compareTo(Item that)
  }
  ```

* Use of the interface requires "Total order"
  * Binary relation ``<=`` that satisfies:
    * Antisymmetry: if ``v <= w`` and ``w <= v`` then ``v = w``.
      * |Question| define antisymmetry relationship.
    * Transitivity: if ``v <= w`` and ``w <= x`` then ``v <= x``.
      * |Question| define transitivity relationship.
    * Totality: either ``v<=w`` or ``w<=v`` or both.
      * |Question| define totality relationship.
    * Examples: numbers, strings, dates.
    * Anti-examples: rock, paper scissors.
  * |Question| What are the 3 conditions that must be satisfied for a relationship to be considered "total order".

* Helper functions:
  * ``less`` is item ``v`` less than ``w``?  
  
    ```
    private static boolean less(Comparable v, Comparable w)
    {
      return v.compareTo(w) < 0;
    }
    ```

  * ``exch`` - swap items in array.

    ```
    private static void exch(Comparable[] a, int i, int j)
    {
        Comparable temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
    ```
