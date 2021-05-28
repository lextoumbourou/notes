---
title: Well-Designed API
date: 2021-04-11 11:00
tags:
  - SoftwareEngineering
---

According to Khorikov, a well-designed API (application programming interface), is one in which [[Observabled Behaviours]] are entirely in the [[Public API]], and implementation details are part of the [[Private API]].

The simplest way to tell if a class leaks a implementation detail into observable behaviour is if number of operations client needs to perform to achieve a goal is > 1.

Good [[Encapsulation]] protects against [[Invariant]] violations, especially as the complexity of the code base increases.

> You cannot trust yourself to do the right thing all the time - so, eliminate the very possibility of doing the wrong thing

[@schellArtGameDesign2015a] *(pg. 100)*

Related to Martin Fowler's [Tell Don't Ask](https://martinfowler.com/bliki/TellDontAsk.html) pattern.