---
title: User Datagram Protocol
date: 2026-09-26 09:00
modified: 2026-09-26 09:00
status: draft
aliases:
- UDP
---

**User Datagram Protocol** (UDP) is a connectionless [Transport Layer](transport-layer.md) protocol that sends datagrams without setting up a connection first.

Unlike [TCP/IP](tcp-ip.md), UDP has no handshake, acknowledgements or retransmission, so it doesn't guarantee delivery or ordering. That makes it lighter and faster, which suits things like video streaming, games and DNS lookups, where losing the odd packet matters less than low latency.

See [The Bits and Bytes of Computer Networking](../reference/moocs/coursera/computer-networking/index.md).
