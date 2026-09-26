---
title: Abstract Data Type
date: 2024-12-22 00:00
modified: 2026-09-26 08:55
status: draft
---

An **Abstract Data Type** (ADT) is a data type defined by its behaviour, meaning the values it can hold and the operations you can perform on it, rather than by how it's implemented.

For example, a [Stack](stack.md) is an ADT defined by its operations (push, pop and peek). It could be implemented with an array or a linked list, and code using the stack shouldn't need to care which. A [Queue](queue.md) is another example.

Separating the interface from the implementation means you can swap implementations (for example, to improve performance) without changing the code that uses them.
