---
title: Roblox Network Ownership
date: 2021-06-24 14:00
tags:
  - Roblox
---

In general, if a part is near an in-game character, its physics will be calculated by that player’s device; otherwise it will be calculated by the server. In either case, the server or client that calculates a part’s physics is called its **owner**.

Since physics updates must be sent over the network, there’s a small delay between when the owner makes the physical changes and when the other devices see those changes. Normally this delay isn’t too noticeable, but issues may occur when part ownership changes.

Roblox let's you set the Network Ownership model 