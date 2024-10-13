---
title: Assertions
date: 2024-02-15 00:00
modified: 2024-02-15 00:00
status: draft
---

**Assertions** in software engineering are checks in code that assert certain true conditions. There are several different types:

## Assertion Types

### [Runtime Assertions](runtime-assertions.md)

Runtime assertions are checked during the program's execution to ensure that certain conditions hold during run time.

For example, `assert x > 0` ensures that x is positive during execution and will terminate if this condition is false.

### [Unit Test Assertions](unit-test-assertions.md)

Assertions in unit tests verify the correctness of specific sections of code. These are not run during runtime, but generally in development and on [Continuous Integration](Continuous%20Integration) infrastructure. The [Test-Driven Development](../../../permanent/test-driven-development.md) methodology ensures the code meets its design and behaves as intended.

For example, a unit test might assert that a function returns the expected value for a given set of input parameters.

### [Compile-time Assertions](compile-time-assertions.md)

These assertions are checked when the code is compiled rather than when run. Use to catch errors as early as possible in the development cycle, often relating to template instantiation or constant expression that the computer can evaluate.

For example, in C++, `static_assert(sizeof(int) == 4)` would ensure that the size of an integer is 4 bytes at compile time.
