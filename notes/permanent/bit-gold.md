---
title: Bit gold
date: 2026-09-26 20:30
modified: 2026-09-26 20:30
status: hidden
summary: Nick Szabo's proposal for scarce digital tokens using proof-of-work, timestamps and distributed ownership records.
tags:
- Bitcoin
- Cryptography
---

**Bit gold** is Nick Szabo's proposal for digital scarcity with minimal reliance on trusted third parties. His December 2005 write-up describes using costly computation to create digital tokens that people could own and transfer [@szaboBitGold2008].

A participant solves a computational challenge to produce a [Proof-of-Work](proof-of-work.md). The result is timestamped and entered into a distributed ownership registry, then becomes the challenge for the next piece of bit gold. A recipient checks the work, timestamp and ownership history [@szaboBitGold2008].

One difficulty is that faster hardware makes new tokens cheaper to produce. Szabo proposed valuing pieces by their creation period and difficulty, then bundling them into units of roughly equal value. His write-up also describes Hal Finney's [RPOW](reusable-proof-of-work.md) as an implemented variant using secure hardware [@szaboBitGold2008].