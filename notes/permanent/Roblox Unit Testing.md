---
title: Roblox Unit Testing
date: 2021-11-16 00:00
tags:
  - Roblox
  - AutomatedTesting
---

Unit testing is a common practice in software. The idea of writing tests for your software is to allow you or a future colleague a way to verify the correctness of the software you wrote. It makes it fast to refactor your code and verify for correctness. You can prove that you've permanelty eliminated a bug for example.

The also help you think about what the code you've written. The process of writing a test for some code is a great way to ensure you fully understand what it's doing.

However, game development requires a lot of prototyping and experimenting, and some code makes little sense to test.

Nonetheless, there are code that is really important to test:

* Code that must be always correct. For example, code that sets up your player's profile data strucutre.
* Code that is complicated or difficult to understand. Code that does calculations, does string parsing and so on.

These also happen to be code that is easy to test.

Somestimes you'll even find benefit in writing the tests before you write code.

However, it's not easy to write unit tests in Roblox. Why?
