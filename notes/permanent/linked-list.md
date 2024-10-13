---
title: Linked List
date: 2023-12-01 00:00
modified: 2023-12-01 00:00
status: draft
tags:
  - ComputerScience
  - DataStructures
---

A **Linked List** is a concrete data structure, where each element containers a reference or *pointer* to the next element in the list.

Each element of a linked list is called **node**, which has a data field and a field stores a reference to the next element. The last element in the list will store a null reference.

A linked list will store a reference to the **head**, which tracks which node is first. If the list is empty, it will point to null.

We can add a new element to the front of the list by creating a new node which points to the current head element. Then, we update the head point to point to this new node.

We can also change the current head by pointing to a different element.

We can add element to the end of a linked list just by pointing the tail element at a new element, which itself points to null.

We can implement a [Stack](stack.md) data structure with a linked list.

To implement a stack with a linked list, we just make the elements of the stack be the nodes in a linked list; every
element storing a value will correspond to a node storing the same value in its data field. The top of the stack is
purely the first node in the linked list. In essence the head pointer in the linked list will point to the top of our
stack.

The order of elements in a stack will then just be reflected by the order of the nodes in the linked list: if an
element is pushed before another in the stack then there will be a pointer from the node corresponding to latter to
the node corresponding to the former. So we have the following picture:

![](../../../_media/linked-list-stack-as-linked-list.png)

## [Doubly Linked List](../../../permanent/doubly-linked-list.md)

A variant of Linked List where nodes also contain a reference to the previous node `x.prev`.
