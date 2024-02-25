---
title: "Week 3: Interfaces, Linked Lists vs. Arrays, and Correctness"
date: 2016-10-04 00:00
category: reference/moocs
status: draft
parent: data-structures-optimizing-performance
---

## Project Overview

1. Create your own LinkedList.
2. Create a Markov Text Generator.

## Abstraction, Interfaces and Linked Lists

* Data abstraction: fundamental concept of programming. Hide implementation details for the user.
* Abstract Data Type (ADT)
  * No implementation.
  * Just defines the methods and properties of a class; a promise.
* Data Structure
  * Actual implementation.
  * Eg ``ArrayList``
* In the real world: data abstraction

## Core: Linked Lists vs Arrays

* ADT: specifies behaviour, not implementation.
  * ArrayList implements the List interface using an array.
    * Provides access to elements in constant time.
    * O(n) to add elements.
  * LinkedList
    * Doubly [Linked List](../../../../permanent/linked-list.md) usually stores a pointer to the head and tail.
    * Each node has a next and prev references.
    * Each node in the list is a ``ListNode`` element.
    * Alternate implementation: Sentinal nodes.
      * Have dummy nodes at start and end of list, which are buffers to keep you running off start and end of the list.
    * O(1) insertion time.
* Core: Generics and Exceptions
* Implementation of ``ListNode``:

          class ListNode<E> {  // not a public class.
              ListNode<E> next;
              ListNode<E> previ;
              E data;  // E == parameterized types: makes ListNode "generic", replace with type when initing it.

  * ``E`` is a paramterized type.
  * Replace with an actual type then refer to type as ``E`` throughout class.
    * Refered to a "generics"
* Handling bad input
  * Since we don't know about type, can't return -1 on bad input.
  * Returning null is a bit yick.
  * Raise an exception: ``throw new NullPointerException("Cannot store null pointers, yo")``
  * If exception is a "checked exception" then we will to declare in method header that it throws the exception.
* Core: Java Code for a Linked List
  * List node defined as follows:

        class ListNode<E> {
            ListNode<E> next; // considered a "recursive class" - uses its own definition in it.
            ListNode<E> prev;
            E data;

              public ListNode(E theData)
              {
                this.data = theData;
              }
        }

  * Linked list class as follows:

        public class MyLinkedList<E>
        {
            private ListNode<E> head;
            private ListNode<E> tail;
            private int size;

            public MyLinkedList() {
                size = 0;
                head = new ListNode<E>(null);
                tail = new ListNode<E>(null);
                head.next = tail;
                tail.prev = head;
            }
        }

## Testing and Correctness

* Core: Testing and Confidence
  * Risk assessment of problem domain should be considered when decided on degrees of confidence for code (self-driving car vs blog).
* Core: Testing Practises
  * Standard Cycle: write code, write tests and test code.
  * Test-Driven Development: write tests, write code and test code.
* Testing types: black box testing and clear box testing.
  * Black box
    * more representative of how users use the code.
    * easier to write by someone unfamiliar with the implementation.

## Core: Markov Text Generation

* High-level idea: walk through text saving a sort of word count of the word at the next position for each word.
* Store in a map with word as the key and next word count as a list.
* Generate text by finding a next word randomly from the word count list for some word count.
