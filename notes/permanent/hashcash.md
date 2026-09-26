---
title: Hashcash
date: 2025-03-09 00:00
modified: 2026-09-26 08:55
status: draft
---


**Hashcash** was an idea proposed by [Adam Back](adam-back.md), as a spam prevention measure, it went on to influence [Proof-of-Work](proof-of-work.md) systems like [Bitcoin](bitcoin.md). The paper was referenced in [Bitcoin: A Peer-to-Peer Electronic Cash System](bitcoin-a-peer-to-peer-electronic-cash-system.md) and some speculate that Adam is [Satoshi Nakamoto](satoshi-nakamoto.md).

The idea is to find a partial hash collision, that is another input string that generates a hash collision on the first n bits.

```
X-Hashcash: 1:20:1303030600:adam@cypherspace.org::McMybZIhxKXu57jd:ckvi
```

The header contains:

* `ver`: Hashcash format version, 1 (which supersedes version 0).
* `bits`: Number of "partial pre-image" (zero) bits in the hashed code.
* `date`: The time that the message was sent, in the format YYMMDD[hhmm[ss]].
* `resource`: Resource data string being transmitted, e.g., an IP address or email address.
* `ext`: Extension (optional; ignored in version 1).
* `rand`: String of random characters, encoded in base-64 format.
* `counter`: Binary counter, encoded in base-64 format.

To prevent the [Double-Spending Problem](double-spending-problem.md), Back suggests storing the hash in a database. Since the date is included as a component on the hash, the database can discard collisions after 2 days.
