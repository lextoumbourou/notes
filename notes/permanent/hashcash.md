---
title: Hashcash
date: 2025-03-09 00:00
modified: 2026-09-26 20:34
status: hidden
summary: Adam Back's proof-of-work system for limiting spam, and how it influenced RPOW and Bitcoin.
tags:
- Bitcoin
- Cryptography
---

**Hashcash** is a [Proof-of-Work](proof-of-work.md) system proposed by [Adam Back](adam-back.md) in 1997 to limit spam and other abuse of internet services. A sender spends computation creating a stamp that the recipient can check cheaply. Back's 2002 paper describes the design and its applications [@backHashcashDenialService2002].

## How it works

The sender tries different counter values until the stamp's hash begins with a required number of zero bits. This is a partial preimage search: the target is a fixed prefix of zeros. The recipient hashes the resulting stamp once to check the work [@backHashcashDenialService2002].

Assuming uniformly distributed hash outputs, a target of 20 zero bits takes about `2^20`, or 1,048,576, attempts on average. Each extra required bit doubles the expected work. The actual number of attempts varies.

## Email stamps

An email stamp can appear in an `X-Hashcash` header. Version 1 has seven colon-separated fields [@backHashcashManual]:

```
X-Hashcash: 1:20:1303030600:adam@cypherspace.org::McMybZIhxKXu57jd:ckvi
```

| Field | Meaning |
| --- | --- |
| `ver` | Format version, here `1`. |
| `bits` | Claimed difficulty, here 20 leading zero bits. |
| `date` | Stamp creation time, formatted as `YYMMDD[hhmm[ss]]` in UTC. |
| `resource` | Intended recipient's email address, or another service identifier. |
| `ext` | Optional extensions, empty in this example. |
| `rand` | Random text that helps different senders generate distinct stamps. |
| `counter` | Text varied while searching for a valid stamp. |

Hashcash stamps use SHA-1 [@finneyReusableProofsOfWork2004]. Hashing the example above, excluding `X-Hashcash: `, produces `00000b7c65ac70650eb8d4f034e86d7d5cd1852f`: exactly 20 leading zero bits.

To reject reuse, the recipient keeps a database of accepted stamps and checks the resource, difficulty and expiry. Expiry is configurable; the command-line tool's documented default is 28 days. Expired entries can then be removed from the database [@backHashcashManual].

## Connection to Bitcoin

Hal Finney's [RPOW](reusable-proof-of-work.md) extended Hashcash by exchanging proofs of work for tokens that could be passed on through successive exchanges [@finneyReusableProofsOfWork2004].

[Satoshi Nakamoto](satoshi-nakamoto.md) cited Back's paper in [Bitcoin: A Peer-to-Peer Electronic Cash System](bitcoin-a-peer-to-peer-electronic-cash-system.md). [Bitcoin](bitcoin.md) applies proof-of-work to blocks of transactions, using the accumulated work in the chain to establish a shared history and make changes to earlier transactions costly [@nakamotoBitcoinPeertoPeerElectronic2008].
