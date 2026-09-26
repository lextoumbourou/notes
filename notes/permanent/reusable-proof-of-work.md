---
title: Reusable Proof-of-Work
date: 2026-09-26 20:30
modified: 2026-09-26 20:30
status: draft
summary: Hal Finney's 2004 system for exchanging Hashcash proofs of work for reusable digital tokens.
tags:
- Bitcoin
- Cryptography
aliases:
- RPOW
- Reusable Proofs of Work
---

**Reusable Proof-of-Work (RPOW)** is a system Hal Finney released in August 2004. It converts [Hashcash](hashcash.md) proofs of work into RSA-signed tokens that can pass between people [@finneyReusableProofsOfWork2004].

Each token can be redeemed only once, but the server exchanges it for a fresh token of equal value. This makes the computational work reusable across transfers without allowing the same token to be spent twice [@finneyReusableProofsOfWork2004].

The server ran inside an IBM 4758 secure cryptographic coprocessor. Remote attestation let users check that it was running the published software. RPOW therefore depended on the server and secure hardware to enforce its rules [@finneyReusableProofsOfWork2004]. [Bitcoin](bitcoin.md) later used a peer-to-peer network and a proof-of-work chain to agree on transaction history [@nakamotoBitcoinPeertoPeerElectronic2008].