---
title: Proof-of-Work
date: 2026-09-26 08:52
modified: 2026-09-26 08:52
status: draft
---

**Proof-of-Work** is a system where one party proves to others that it has spent a certain amount of computational effort, in a way that is expensive to produce but cheap to verify.

Typically, the work is finding an input whose [Hash Function](hash-function.md) output has some property, like starting with a certain number of zero bits. The only way to find one is by brute-force trial and error, but anyone can check the answer by computing a single hash.

[Hashcash](hashcash.md) used it to make sending spam email expensive. [Bitcoin](bitcoin.md) uses it to secure the blockchain: miners compete to find a nonce that makes a block's hash fall below a difficulty target, and the chain with the most accumulated work is treated as the valid one. See [Bitcoin: A Peer-to-Peer Electronic Cash System](bitcoin-a-peer-to-peer-electronic-cash-system.md).
