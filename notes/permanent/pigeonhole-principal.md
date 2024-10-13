---
title: Pigeonhole Principal
date: 2023-12-23 00:00
modified: 2023-12-23 00:00
status: draft
aliases:
  - Dirichlet Drawer Principle
---

A obvious little principle that turns out to be quite useful.

If I have more pigeons than I do pigeonholes

Then there will be *at least one* pigeonhole with *more than one* pigeon in it.

That obvious principle can be generalised by substitute objects and boxes for pigeons and pigeonholes.

Now I can express the general version of this in formal maths-y language:

If there are $k+1$ or more objects to be placed in k boxes, there is at least one box containing more than one object.

Also known as [Dirichlet Drawer Principle]().

Examples:

* "If I 4 bedrooms and 5 children, then some of the children will have to share a room"
* "If 5 cards are drawn from a standard deck of 52 cards, then at least 2 of them are of the same suit"
* "If there are more than 368 people at a venue, in a class, at least 3 of them have will the same bday."

---

## [[Generalised Pigeonhole Principle]]

Generalised principle: if there are $N$ objects to be placed in $k$ boxes, there is at least one box containing at least $[N/k]$ (ceiling) objects.

Can be proved using [Contradiction](logical-contradiction.md).

* Assume no box has more than $[N/k]-1$ objects
* Number of objects $\le k([N/k] - 1) < k(N/k+1-1)=N$
* Number of objects < N
* This is a contradiction, as we have N objects.

---

Here's an example of this problem:

How many cards from a standard deck of 52 cards must be selected to guarantee that 3 cards are from the same suit?

There are four suits. If we pick 8 cards evenly spread, we have 2 cards from all suits. Now, any further card we pick after that, will ensure that we have 3 cards in the same suit.

So the answer is 9.

We can verify this using the generalised principle: [N/k] substituting k for 4 boxes (4 suits) and number of objects for 3: $[N/4] = 3$ and confirm $[9/4] = 3$.
