---
title: Permutation
date: 2023-04-11 00:00
modified: 2023-04-11 00:00
status: draft
---

A permutation is an **ordered** collection of objects, unlike a [Combination](combination.md) which is unordered.

A common mnemonic to remember the different is to think that the $P$ in Permutations stands for position:

**P**ermutations: **P**osition mattters.
**C**ombinations: **C**hoose without concern for order.

An ordered subset of $r$ objects is called an *r-permutation*.

The set of r-permutation from a set of $n$ elements is written as:

$P(n, r)$

If I have a set of 3 elements: $\{a, b, c\}$, and I want to find all the permutations of size 2, I can write it as follows:

$P(3, 2)$

Let me list all the permutations:

{a, b}
{a c}
{b c}

{b, a}
{c a}
{c a}

The formula for counting permutations is: 

$P(n, r) = \frac{n!}{(n - r)!}$

Given that n is positive, and r is an integer where r \leq n.

3! = 3 x 2 x 1 = 6
3-2 = 1! = 1

6/1 = 6

## Permutation with repetition

Consider this problem:

How many different 4-letter sequences can be made using the letters in
the word "door"? You can only use each letter once, noting that the letter
’o’ appears twice and can therefore be used twice. 


Since o appears twice, and it does not matter what order we write the 0s in, how do we account for that?

We can use this formula: 

n!/x_1!, x_2!, x_3!

Where x is the number of times a letter is repeated.

d = 1
o = 2
r = 1

4! / 1!2!1! = 24 / 1 * 2 * 1 = 12



