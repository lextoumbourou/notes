---
title: Foundation Maths
date: 2023-04-18 00:00
status: draft
---

## 14. Number bases

* The decimal system
    * Aka base 10.
    * $253 = 200 + 50 + 3 = 2(100) + 5(10) + 3(1) = 2(10^2) + 5(10^1) + 3(10^0)$
    * 10 digits: 0, 1, ... 9
    * Denote numbers using small subscript: $253_{10}$
* The binary system
    * Base 2
    * Converting from binary to decimal:
        * $110101_2 = 1(2^5) + 1(2^4) + 0(2^3) + 1(2^2) + 0(2^1) + 1(2^0)$
* Examples
    * Convert $83_{10}$ to binary:
        * $2^6 + 2^4 + 2^1 + 2^0$
        * $1010011_2$
    * Express $200_{10}$ as binary:
        * $2^7 + 2^6 + 2^3$
        * $11001000_2$
* Another way to convert decimal number to binary is to divide by 2 repeatedly and note remainder.
    * 83:
        * 83 % 2 = 41 r 1
        * 41 % 2 = 20 r 1
        * 20 % 2 = 10 r 0
        * 10 % 2 = 5 r 0
        * 5 % 2 = 2 r 1
        * 2 % 2 = 1 r 0
        * 1 % 2 = 0 r 1
        * Then write out the remaining from bottom to top:
            * 1010011
* Octal system
    * 8 base
* Hexidecimal
    * 16 base: 0, 1, ... 9, A, B, ..., F
