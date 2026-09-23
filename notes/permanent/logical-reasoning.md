---
title: Logical Reasoning
date: 2025-09-09 00:00
modified: 2026-09-24 08:32
summary: "Describing the 3 types of logical reasoning: Abduction, deduction and induction."
cover: /_media/logical-reasoning-wooden-puzzle-kieran-wood.jpg
cover_credits: Photo by <a href="https://unsplash.com/@kieran_wood">Kieran Wood</a> on <a href="https://unsplash.com/photos/brown-wooden-blocks-on-black-wooden-table-UOk1ghQ7juY">Unsplash</a>
tags:
- SymbolicAI
- LogicalReasoning
---

**Logical Reasoning** refers to the process of arriving at conclusions based on some premises. The premises and conclusions are [Propositions](propositions.md) - statements that are either true or false. It is the foundation of Symbolic AI, which makes inferences using logic and knowledge-based systems.

There are three main types of logical reasoning:

- [Abduction](#abduction): the conclusion is a plausible explanation, but not necessarily a fact.
- [Deduction](#deduction): the conclusion is guaranteed to be correct if the argument is valid and all its premises are true.
- [Induction](#induction): a generalisation based on data that remains open to revision. Machine learning is primarily based on inductive reasoning.

## Abduction

Abduction, or abductive reasoning, refers to finding a plausible explanation for observations, often choosing the best explanation among alternatives. The conclusion is not necessarily correct, only a plausible explanation.

One simple pattern is:

$$
\begin{aligned}
\text{Rule:} &\quad P \implies (\text{implies}) \ Q \\
\text{Observation:} &\quad Q \\
\text{Possible explanation:} &\quad P
\end{aligned}
$$

- Rule: All men are mortal.
- Observation: Socrates is mortal.
- Possible explanation: Socrates is a man.

The rule and observation alone do not establish that Socrates is a man. Other living things are mortal too.

Much of Sherlock Holmes' "deductive method" is better described as abductive reasoning: Holmes observes clues (effects) and infers the most likely cause or explanation for those observations.

## Deduction

On the other hand, deduction, or deductive reasoning, draws a conclusion that must follow from the premises. The conclusion is guaranteed to be correct if the argument is valid and all its premises are true.

One common pattern is:

$$
\begin{aligned}
\text{Rule:} &\quad P \implies (\text{implies}) \ Q \\
\text{Observation:} &\quad P \\
\text{Conclusion:} &\quad Q
\end{aligned}
$$

- Rule: All men are mortal.
- Observation: Socrates is a man.
- Conclusion: Socrates is mortal.

In this example, deduction moves from the general to the specific. What defines deduction, though, is that the conclusion must follow if the premises are true.

## Induction

Finally, induction takes a set of facts and aims to build a general conclusion that may or may not be true. Machine learning is primarily based on inductive reasoning, though modern ML systems may also incorporate deductive and abductive elements.

One common pattern is: specific observations → general rule or pattern.

Example:

- Fact: Socrates is a man.
- Fact: Socrates is mortal.
- Fact: Plato is a man.
- Fact: Plato is mortal.
- Fact: Aristotle is a man.
- Fact: Aristotle is mortal.
- Possible rule: All men are mortal.

Inductive reasoning often moves from specific observations to general principles. The strength of the conclusion depends on the number, variety and representativeness of the observations. Unlike valid deduction from true premises, inductive reasoning provides support rather than certainty. Further testing or observation can strengthen or undermine the conclusion, but does not guarantee it.
