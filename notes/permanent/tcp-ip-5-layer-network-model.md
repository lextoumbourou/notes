---
title: "TCP/IP Five-Layer Network Model"
date: 2025-03-08 00:00
modified: 2026-09-26 08:55
status: draft
---

The **TCP/IP Five-Layer Network Model** splits network communication into five layers:

1. Physical
2. Data Link
3. [Network Layer](network-layer.md)
4. [Transport Layer](transport-layer.md)
5. [Application Layer](application-layer.md)

It's a hybrid between the original four-layer TCP/IP model and the seven-layer [OSI Model](osi-model.md). Each layer uses the services of the layer below it: for example, [HTTP](http.md) at the application layer runs on top of TCP at the transport layer, which runs on top of IP at the network layer.

See [TCP/IP](tcp-ip.md).
