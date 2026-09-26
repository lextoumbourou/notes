---
title: Simplified Payment Verification
date: 2025-03-15 00:00
modified: 2026-09-26 08:55
status: draft
---

**Simplified Payment Verification** (SPV) is a way to verify Bitcoin payments without running a full network node, described in section 8 of [Bitcoin: A Peer-to-Peer Electronic Cash System](bitcoin-a-peer-to-peer-electronic-cash-system.md).

An SPV client only keeps the block headers of the longest chain. To check a transaction, it gets the [Merkle Tree](merkle-tree.md) branch linking the transaction to a block, which shows the network has accepted it, and further blocks added on top confirm it.

The trade-off is that it relies on the network being honest: verification is only reliable as long as honest nodes control the network.
