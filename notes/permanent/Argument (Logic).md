---
title: Argument
date: 2023-01-03 00:00
status: draft
---

An argument in [[Propositional Logic]] is a sequence of [[Proposition]]s.

The final argument is the [[Conclusion]] and the other arguments are called premise or hypotheses.

An argument is a [[Valid Argument]] if the truth of all premise implies the conclusion.

Example

If it's cold, I'll go to bed early.
It's cold.
- - -
Therefore, I'll go to bed early.

The argument is valid since all the premise are true.

Here's an invalid argument:

If it's hot, I'll drink water.
I'll drink water

- - -

Therefore, it's hot.

Why? Because there can be one example where all the premise can be true, but the conclusion false.

there are situations where premises are true and conclusion is false.

It it's cold, then the statement "If it's hot, I'll drink water" is true by default.
and the statement i'll drink water is true

But here the conclusion is false. It is cold, not hot!

Let's build a truth table to understand:

| It's hot | I'll drink water | It's hot -> I'll drink the water |
| -------- | ---------------- | -------------------------------- |
| False    | False            | True                             |
| **False**    | **Truth**            | **True**                             |
| True     | False            | False                            |
| True     | True             | True                                 |

See [[Rules of Inference]].