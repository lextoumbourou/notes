---
title: "Bitcoin"
slug: bitcoin
date: 2024-10-05 00:00
modified: 2026-09-26 20:34
status: hidden
category: paper
summary: A peer-to-peer cash system.
paper_title: "Bitcoin: A Peer-to-Peer Electronic Cash System"
paper_url: https://bitcoin.org/bitcoin.pdf
paper_authors: Satoshi Nakamoto
paper_year: 2008
tags:
- Bitcoin
- Cryptography
aliases:
- Bitcoin Paper
- Bitcoin
---

The paper describes a new type of peer-to-peer electronic cash that allows payments to be sent without a financial institution. Bitcoin utilises [Digital Signatures](digital-signatures.md) and solves [Double-Spending Problem](double-spending-problem.md) by utilising hash-based [Proof-of-Work](proof-of-work.md).

The longest chain - the ledger - of blocks serves as proof of the sequence of events, and also proof that it came from the largest pool of CPU power.

As long as the majority of CPU power is controlled by nodes that are not cooperating to attack the network, they will generate the longest chain and outpace attackers [@nakamotoBitcoinPeertoPeerElectronic2008].

## Predecessors

Earlier work on proof-of-work and electronic cash includes:

* [Hashcash](hashcash.md) by Adam Back, cited in the Bitcoin whitepaper [@nakamotoBitcoinPeertoPeerElectronic2008].
* [B-Money](b-money.md) by Wei Dai, also cited in the whitepaper [@nakamotoBitcoinPeertoPeerElectronic2008].
* [Bit gold](bit-gold.md) by Nick Szabo, a proposal combining proof-of-work, timestamps and distributed ownership records [@szaboBitGold2008].
* [RPOW](reusable-proof-of-work.md) by Hal Finney, a working 2004 system that exchanged Hashcash for transferable tokens, using a server with secure hardware [@finneyReusableProofsOfWork2004].

## Related

* [Satoshi Nakamoto](satoshi-nakamoto.md)
* [Can Claude Opus 5.5 find any new leads on Satoshi Nakamoto?](can-claude-opus-5-5-find-any-new-leads-on-satoshi-nakamoto.md)
