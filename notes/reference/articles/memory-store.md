---
title: Memory Store
date: 2021-09-21 13:30
status: draft
tags:
  - Roblox
---

* MemoryStoreService provides fast in-memory storage accessible across servers.
    * Supports 2 data structures: queues and sorted maps
* Provides low latency, high throughput access.
* Useful for:
    * Global leaderboards
    * Skill-based matchmaking queues
    * Auction houses
* Limits
    * Memory size quota:
        * 64KB + 1KB * num users
    * Use explicity deletion or expiration to ensure items don't remain indefinitely.
* Queues
    * Queues mainain a first-in-first-out (FIFO) sequence.
    * You can also set a priority when adding items to the qqueue.