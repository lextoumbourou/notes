---
title: Modular Arithmetic
date: 2024-12-17 00:00
modified: 2026-09-26 08:55
status: draft
---

**Modular Arithmetic** is a system of arithmetic for integers where numbers "wrap around" after reaching a certain value, called the modulus.

The classic example is a 12-hour clock: 4 hours after 10 o'clock is 2 o'clock, so $10 + 4 \equiv 2 \pmod{12}$.

We say $a \equiv b \pmod{n}$ ($a$ is congruent to $b$ modulo $n$) if $a$ and $b$ have the same remainder when divided by $n$.

It's the foundation for a lot of cryptography, including [RSA](rsa.md). The [Extended Euclidean Algorithm](extended-euclidean-algorithm.md) is used to find modular inverses.
