---
title: Well-Designed API
date: 2021-04-11 00:00
tags:
  - SoftwareEngineering
---

According to Khorikov, a well-designed API (application programming interface) is one in which observable behaviours are entirely in the public API, and implementation details are part of the private API.

The simplest way to tell if a class leaks an implementation detail into observable behaviour is if the number of operations the client needs to perform to achieve a goal is > 1.

Good [Encapsulation](encapsulation.md) protects against invariant violations, especially as the complexity of the code base increases.

> You cannot trust yourself to do the right thing all the time - so eliminate the very possibility of doing the wrong thing

[@schellArtGameDesign2015a] *(pg. 100)*

It is related to Martin Fowler's [Tell Don't Ask](https://martinfowler.com/bliki/TellDontAsk.html) pattern.
