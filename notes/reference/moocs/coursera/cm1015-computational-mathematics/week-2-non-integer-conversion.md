---
title: "Week 2 - Non-Integer Numbers Conversion"
date: 2023-04-20 00:00
modified: 2023-04-20 00:00
status: draft
---

* In [Week 1 - Number Bases](week-1-number-bases.md) we saw how the to get integers from one base to another, in this lecture learn how to convert decimals.
    * Example in base 10:
        * $17.375_{10} = 1 \times 10^{1} + 7 \times 10^{0} + 3 \times 10^{-1} + 7 \times 10^{-2} + 5 \times 10^{-3}$
        * $= 10 + 7 + 3/10 + 7/100 + 5/1000$
        * Numbers to the right of the decimal point are negative exponents.
    * Base 10 to Binary
        * Separate integer part 17 and fractional part 0.375
            * $17_{10} = 10001_2$
            * $0.375 \times 2 = 0.75 = 0 + 0.75$
            * $0.75 \times 2 = 1.5 = 1 + 0.5$
            * $0.5 \times 2 = 1.0 = 1 + 0$ (stop at 0)
            * Read from top-to-bottom (unlink integer conversion)
            * Result: $0.375_{10} = 0.011_{2}$
    * Binary to Base 10
        * Example: $1101.101_2 = 1 \times 2^3 + 1 \times 2^{2} + 0 \times 2^{1} + 1 \times 2^{0} + 1 \times 2^{-1} + 0 \times 2^{-2} + 1 \times 2^{-3}$
        * $= 8 + 4 + 1 + 1/2 + 1/8 = 13.625_{10}$
    * General rule:
        * $a_n + a_{n-1} + a_{n-2} ... a_{0} . c_{-1} c_{-2} ... c_{-k}$ in base b.
    * In decimal units:
        * $a_n \times b^{n} + a_{n - 1} \times b^{n-1} + .... + a_0 \times b^0 +$
    * $c_{-1} \times b^{-1} + c_{-2} \times b^{-2} + ... c_{-k} \times b^{-k}$
* Practice questions
    * $11.625$ in binary
        * $11_{10}$
            * $11 / 2 = 5 \ r \ 1$
            * $5 / 2 = 2 \ r \ 1$
            * $2 / 2 = 1 \ r \ 0$
            * $1 / 2 = 0 \ r \ 1$
            * $1011$
        * $0.625$
            * $0.625 \times 2 = 1 + .25$
            * $0.25 \times 2 = 0 + 0.5$
            * $0.5 \times 2 = 1 + 0$
            * $.101$
        * $1011.101$
    * $1/64$ in binary
        * $0.015625$
            * $.015625 \times 2 = 0 + 0.03125$
            * $0.03125 \times 2 = 0 + 0.0625$
            * $0.0625 \times 2 = 0 + 0.125$
            * $0.125 \times 2 = 0 + 0.25$
            * $0.25 \times 2 = 0 + 0.5$
            * $0.5 \times 2 = 1 + 0$
            * $0.000001$
* Operations with binary numbers
    * You can convert numbers into the base you want to perform operations on them.
    * Addition:
        * $101_2 + 111_2$
        * Similar to decimal sum in column divide y two carry over quotient and take remainder
     * Subtraction
         * In base 10 subtraction, you "borrow" from the adjacent left column.
         * Same in base 2, except you get a 2 from the adjacent column.
           ![](_media/week-2-non-integer-conversion-base2-subtraction.png)
     * Multiplication
         * Same as decimal
           ![](/_media/week-2-non-integer-conversion-multiplication.png)
   * Division
      ![](/_media/week-2-non-integer-conversion-division.png)
  * Need to revise this.
