---
title: Deterministic Backoff
date: 2024-10-14 00:00
modified: 2024-10-14 00:00
status: draft
---

**Deterministic Backoff** is an algorithm used in network communication where devices follow a predefined, fixed sequence of wait times to avoid collisions when attempting to access a shared medium. Each device's backoff time is predetermined, ensuring that collisions are systematically avoided without random delays.

Unlike [Binary Exponential Backoff](binary-exponential-backoff.md) (BEB), which uses random and exponentially increasing backoff times after collisions, Deterministic Backoff follows a fixed pattern, offering more predictability but less flexibility in dynamic network conditions.
