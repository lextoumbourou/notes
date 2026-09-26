---
title: Subnetting
date: 2026-09-26 09:00
modified: 2026-09-26 09:00
status: draft
---

**Subnetting** is the process of splitting a large IP network into smaller networks called subnets.

A subnet mask marks which bits of an IP address identify the network (and subnet) and which identify the host. For example, the mask 255.255.255.0 leaves the last 8 bits for hosts, giving 256 addresses, 254 of which can be assigned (the first and last are reserved for the network and broadcast addresses).

Routers on the wider internet only need the network ID to get traffic to the right place, and routers inside the network use the subnet and host IDs to deliver it. [CIDR](cidr.md) replaced the old class-based system with a more flexible way of writing and allocating these ranges.

See [The Bits and Bytes of Computer Networking](../reference/moocs/coursera/computer-networking/index.md).
