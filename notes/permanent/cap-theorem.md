---
title: CAP Theorem
aliases: "Brewer's Theorem"
date: 2021-04-03 00:00
tags: 
  - DataEngineering
---

Also known as Brewer's theorem, after computer scientist Eric Brewer, is the assertion that a distributed data store can't provide 3 of the following capabilities simultaneously:

* Consistency: every read request gets the most recent write
* Availability: Every read request receives a non-error response
* Partition tolerance: The system continues to operate despite some messages dropped between nodes

Though often referred to as a "two out of three" tradeoff (you can have 2 out of 3 properties), it's more of a question about how the datastore deals with partitional intolerance: does it return errors, or does it return out-of-date data?

## Examples

### HBase

Guarantees that each read request will get the most recent write. However, if a region is unavailable, read requests will fail.

### PostGres HA

Postgres in a leader/follower configuration provide availability but not consistency. The follower will often give an out-of-date copy of data.

---

* [CAP theorem (Wikipedia)](https://en.wikipedia.org/wiki/CAP_theorem)
