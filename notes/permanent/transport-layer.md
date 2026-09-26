---
title: Transport Layer
date: 2025-03-08 00:00
modified: 2026-09-26 08:55
status: draft
---

The **Transport Layer** is the layer of the [TCP/IP Five-Layer Network Model](tcp-ip-5-layer-network-model.md) that provides end-to-end communication between processes running on different hosts, using port numbers to tell processes apart.

The two main protocols are TCP, which is connection-oriented and guarantees reliable, ordered delivery, and [UDP](user-datagram-protocol.md), which is connectionless and faster but doesn't guarantee delivery.

It sits between the [Network Layer](network-layer.md) and the [Application Layer](application-layer.md), and corresponds to layer 4 of the [OSI Model](osi-model.md).
