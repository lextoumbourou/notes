---
title: Week 6 - Modular Arithmetic Continued
date: 2023-05-22 00:00
modified: 2023-05-22 00:00
status: draft
---

## Lesson 3.2 - Operations with congruent numbers

* Addition and subtraction (mod k): map number in "minimal subset" called $Min_k$, then sum or subtract. Adjust if result is not in $Min_k$
    * Example (mod 12): mapping to min_12 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
        * $14 + 28 \equiv 2 + 4 \equiv 6 \rightarrow 14 + 28 = 6 \text{(mod 12)}$
        * $28 - 14 \equiv 4 - 2 = 2 -> 28 - 14 = 2 (mod 12)$
        * $11 + 28 \equiv 11 + 4 = 15 \equiv 3 \rightarrow 11 + 28 \equiv 3 (mod 12)$
* Multiplication with modular arithemtic.
    * First map to minimal subset and multiply.
    * If it still doesn't belong, you can remap.
* Division with modular arithmetic
    * Division more complex.
    * If you had 4/12 (mod 6), it's not defined. As 12 is congruent to 0.
    * So, you first calculate [[Multiplicative Inverse]].
        * Multiplicative inverse of $m^{-1}$ of integer m: $m \times m^{-1} = 1 \text{(mod k)}$
        * Then define a/b (mod k) as $a \times b^{-1}$
        * Then you just apply multiplication rule.
