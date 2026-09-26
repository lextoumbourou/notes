---
title: Network Address Translation
date: 2026-09-26 09:00
modified: 2026-09-26 09:00
status: draft
aliases:
- NAT
---

**Network Address Translation** (NAT) is a technique where a router rewrites the source or destination IP addresses of packets as they pass through it, usually so that many devices on a private network can share a single public IP address.

In the common one-to-many setup (also called IP masquerading), the router replaces each internal device's private IP with its own public IP on outbound traffic. To send return traffic to the right device, it keeps track of connections by port, using techniques like port preservation, and port forwarding can be configured so that traffic to a given port always goes to a particular internal host.

NAT sits at the [Network Layer](network-layer.md) but relies on [Transport Layer](transport-layer.md) ports to work in practice.

See [The Bits and Bytes of Computer Networking](../reference/moocs/coursera/computer-networking/index.md).
