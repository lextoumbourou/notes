---
title: How to time-stamp a digital document
date: 2025-03-15 00:00
modified: 2026-09-26 08:55
status: draft
---

**How to Time-Stamp a Digital Document** is a 1991 paper by Stuart Haber and W. Scott Stornetta, published in the Journal of Cryptology.

It proposes ways to certify when a digital document was created or last changed, so that nobody, not even the timestamping service, can backdate or forward-date it. The main idea is to link each new timestamp to the hash of the previous one, forming a chain, so an old timestamp can't be changed without breaking everything that came after it.

The paper is cited in [Bitcoin: A Peer-to-Peer Electronic Cash System](bitcoin-a-peer-to-peer-electronic-cash-system.md), and the chain of hashed timestamps is an ancestor of the blockchain. It was followed up by [Improving the efficiency and reliability of digital time-stamping](improving-the-efficiency-and-reliability-of-digital-time-stamping.md).
