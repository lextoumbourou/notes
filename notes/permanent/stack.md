---
title: Stack
date: 2023-11-19 00:00
modified: 2023-11-19 00:00
tags:
  - Algorithms
  - ComputerScience
summary: a data structure in which we can only read and write from the top
cover: /_media/stack-cover.png
hide_cover_in_article: true
---

A **stack** is a variable length data structure that supports last in, first out access, and only the **top** element is accessible.

![Diagram of a Stack](../_media/stack-diagram.png)

Think of a stack of plates - you can only add and retrieve from the top.

A stack supports the following operations in its typical form:

<table class="table-border">
    <tr>
        <th>Operation</th>
        <th>Pseudocode</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><code>push(o)</code></td>
        <td><code>PUSH(o, s)</code></td>
        <td>Place <code>o</code> on top of stack.</td>
    </tr>
    <tr>
        <td><code>top</code></td>
        <td><code>TOP(s)</code></td>
        <td>Return the element at top of stack.</td>
    </tr>
    <tr>
        <td><code>pop!(o)</code></td>
        <td><code>POP(s)</code></td>
        <td>Removes and returns the element at the top of the stack.</td>
    </tr>
    <tr>
        <td><code>empty?</code></td>
        <td><code>EMPTY(s)</code></td>
        <td>Return True or False if empty.</td>
    </tr>
    <tr>
        <td><code>Construct new(empty) stack</code></td>
        <td><code>new STACK s</code></td>
        <td>Create a new stack.</td>
    </tr>
</table>
