---
title: Server Authoritative Multiplayer
date: 2021-06-24 00:00
cover: /_media/hal-9000.png
tags:
  - MultiplayerArchitecture
---

A typical multiplayer game architecture where the server has authority over the game state. The server keeps track of players' positions, the resources they own, scores, etc., and validates any changes the client requests to the game.

It's a design decision made over relayed multiplayer, or [Client Authoritative](Client Authoritative), where one client has authority and reconciles all game messages.

The trade-off with this architecture is that it introduces a latency penalty for every game action, as clients must pass requests through the server. Game engines like the one in Roblox work around this by allowing clients to modify certain parts of the state without server permission, like their own character movement and movement of some game objects, etc.

The diagram below shows a hypothetic request to update a client's score.

![Server Authoritative Multiplayer](/_media/server-auth.png)

In this architecture, the server act's as the games [Rule Enforcer](rule-enforcer.md).

#### Reference

* [Nakama Server - Authoritative Multiplayer](https://heroiclabs.com/docs/gameplay-multiplayer-server-multiplayer/)
* [Roblox Client-Server Model](https://developer.roblox.com/en-us/articles/Roblox-Client-Server-Model)
