---
title: Server Authoritative Multiplayer
date: 2021-06-24 20:30
cover: /_media/hal-9000.png
tags:
  - MultiplayerArchitecture
---

Server Authoritative Multiplayer is a game architecture where the server has authority over the state of a game. The server keeps track of players' positions in the game world, the resources they own, scores, etc. The server validates any changes clients request to the game state.
 
It's a design decision made over relayed multiplayer, or client authoritative, where one client has authority and reconciles all game messages.
 
The trade-off with this architecture is that it introduces a latency penalty for every game action, as clients must pass requests through the server. Game engines like Roblox's allow clients to modify certain parts of the state without server permission, like their own character movement and movement of some game objects, etc.
 
In this architecture, the server act's as the games [[Rule Enforcer]].
 
![Server Authoritative Multiplayer](/_media/server-auth.png)

#### Reference
  
* [Nakama Server - Authoritative Multiplayer](https://heroiclabs.com/docs/gameplay-multiplayer-server-multiplayer/)
* [Roblox Client-Server Model](https://developer.roblox.com/en-us/articles/Roblox-Client-Server-Model)