---
title: Merkle Tree
date: 2025-03-15 00:00
modified: 2025-03-15 00:00
summary: a data structure where each node contains the hash of its child nodes
cover: /_media/merkle-tree.png
hide_cover_in_article: true
tags:
- DataStructures
- Cryptography
---

**Merkle Tree** is a cryptographic data structure where each non-leaf node contains the hash of its child nodes, which allows for efficient verification of the integrity of large datasets without requiring the entire dataset.

Leaf nodes contain hashes of individual data blocks, while parent nodes contain hashes of their children's hashes.

![merkle-tree.png](../_media/merkle-tree.png)

*Original illustration by David Göthberg. [Source](https://commons.wikimedia.org/wiki/File:Hash_Tree.svg)*

Merkle Trees enable "Merkle proofs" to verify that a specific transaction or piece of data is included in a dataset by only checking a small number of hashes rather than the entire set.

In [Bitcoin](../../../permanent/bitcoin.md), the blockchain uses Merkle Trees to save disk space by hashing transactions into a tree structure with only the Merkle root stored in the block header, which enables Simplified Payment Verification (SPV), allowing lightweight clients to verify transactions without downloading the entire blockchain.

![merkle-tree-in-bitcoin.png](../_media/merkle-tree-in-bitcoin.png)

**Diagram from [Bitcoin: A Peer-to-Peer Electronic Cash System](https://bitcoin.org/bitcoin.pdf)*

Merkle Trees are also used in:

* **Git**: For efficient repository versioning.
* **IPFS**: For content-addressed storage.
* **Ethereum**: For tracking state changes and transaction verification.
