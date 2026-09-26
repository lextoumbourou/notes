---
title: CIDR
date: 2026-09-26 09:00
modified: 2026-09-26 09:00
status: draft
aliases:
- Classless Inter-Domain Routing
---

**CIDR** (Classless Inter-Domain Routing) is a way of allocating and describing IP address ranges that replaced the old Class A, B and C address system.

Instead of fixed class boundaries, CIDR writes an address with a suffix giving the number of bits used for the network, e.g. 9.100.100.100/24 means the first 24 bits are the network and the remaining 8 bits are for hosts. This lets networks be sized to fit, rather than jumping between a class C (254 hosts) and a class B (65,534 hosts).

CIDR notation is also used for IPv6. See [Subnetting](subnetting.md) and [The Bits and Bytes of Computer Networking](../reference/moocs/coursera/computer-networking/index.md).
