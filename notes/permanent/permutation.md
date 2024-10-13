---
title: Permutation
date: 2023-04-11 00:00
modified: 2023-04-11 00:00
status: draft
tags:
  - Combinatorics
  - DiscreteMath
summary: an ordered collections of objects
---

A permutation is an **ordered** collection of objects.

If I have 5 people and I want to know how many different ways they can be arranged in a team of 5:

1. I have 5 choices for the first slot.
2. For the 2nd slot, only 4 choices, since I've chosen one in the first.
3. For the 3rd, only 3 choices.
4. For the 4th, only 2 choices.
5. For the 5th, only 1 choice.

So the total number of choices is: 5 x 4 x 3 x 2 x 1 = 5! = 120

Order matters for permutation, which differentials it from a [Combination](combination.md), which is unordered collection of objects.

A useful mnemonic to remember the difference is to think that the $P$ in Permutations stands for position:

**P**ermutations: **P**osition matters.
**C**ombinations: **C**hoose without concern for order.

## [R-Permutation](r-permutation.md)

An ordered subset of $r$ objects is called an r-permutation.

For example, if I had 25 people to choose from but only 5 slots on the team.

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

Given that n is positive, and r is an integer where $r \leq n$.

$3! = 3 x 2 x 1 = 6$
$3-2 = 1! = 1$

$6/1 = 6$

## [Permutation with Repetition](../../../permanent/permutation-with-repetition.md)

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
