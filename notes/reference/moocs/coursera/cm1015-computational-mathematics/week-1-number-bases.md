---
title: Week 1 - Number Bases
date: 2023-04-18 00:00
modified: 2023-04-18 00:00
status: draft
category: reference/moocs
parent: cm1015-computational-mathematics
---

## 1.1 Introduction to number bases: conversion to decimal

* Number Bases
    * Word "digit" comes from Latin word for finger.
        * Base 10 system most familiar to us as we have 10 fingers.
    * Sexagesimal System based on 60 digits
        * Introduced by the Sumerians and Babylonians.
        * Still used for angles, time and geographical coordinates.
    * Base 10
        * When you count after 9 you go back to 0 and add digit to left starting with 1.
        * Largest numbers:
            * With 1 digit: 9
            * With 2 digits: 99
            * With 3 digits: 999
    * Base 2
        * Uses digits 0 and 1.
        * When > 0 and the right-most digit is 0, it's a power of 2.
            * 10 = 2, 100 = 4, 1000 = 8
    * Generic base $n$
        * Use digits $0, 1, n - 1$
        * The maximum single number that can be used is `n - 1`
    * Decomposing number bases:
        * $127_{10} = 1 \times 10^2 + 2 \times 10^1 + 7 \times 10^0$
    * Hexadecimal System
        * Base 16 system.
        * 16 digits: 0, 1, ..., 9, A, B, C, D, E, F
        * $1F = 1 \times 16^1 + 16 \times 16^0 = 31_{10}$
    * Generic base b, conversion to decimal
        * If you have $a_n, a_n-1, a_n-2, ..., a_0$ and want to convert to base b:
            * $a_n \times b^{n} + a_{n-1} \times b^{n-1} +  ... + a_0 \times b^{0}$
* Reading
    * Croft, A. and R. Davison Foundation maths. (Harlow: Pearson, 2016) 6th edition. Chapter 14 Number bases.

### 1.1 Number bases: decimal to binary conversion

* Repeated Division
    * Method for converting decimal to another base
    * Example: Convert $58_{10}$ in base 2.
        * 58 / 2 = 29 r 0$
        * 29 / 2 = 14 r 1
        * 14 / 2 = 7 r 0
        * 7 / 2 = 3 r 1
        * 3 / 2 = 1 r 1
        * 1 / 2 = 0 r 1
        * Then work backwards, any remainder is a 1, else 0
            * $111010_2$
    * Example: Convert 558_10 in base 5
        * 558 / 5 = 111 r 3
        * 111 / 5 = 22 r 1
        * 22 / 5 = 4 r 2
        * 4 / 5 = 0 r 4
        * Result: $4213_5$
    * Example: Convert 639 to binary
        * 639 / 2 = 319 r 1
        * 319 / 2 = 159 r 1
        * 159 / 2 = 79 r 1
        * 79 / 2 = 39 r 1
        * 39 / 2 = 19 r 1
        * 19 / 2 = 9 r 1
        * 9 / 2 = 4 r 1
        * 4 / 2 = 2 r 0
        * 2 / 2 = 1 r 0
        * 1 / 2 = 0 r 1
        * Result: 1001111111
