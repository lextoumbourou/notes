---
title: Lua Coroutines
date: 2021-01-18 00:00
tags:
  - Lua
---

Coroutines in Lua are related to [Generators (Python)](Generators%20(Python)). They allow a function to be written to return intermittantly using the `yield()` function call on the `coroutine` table. The coroutine is created by passing a function to the `coroutine.create(f)` function. It returns a value of type `thread`, which can then be passed to `coroutine.resume(t)`, which will run the code until the next `coroutine.yield` call or `return`.

Calls to `resume` are run in [protected mode (Lua)](protected%20mode%20(Lua)), which means errors won't be raised, but returned as a string to the caller.

The Roblox API includes a `coroutine.wrap(f)` function, which returns a function that can be called without requiring `resume`.

References:

* https://developer.roblox.com/en-us/api-reference/lua-docs/coroutine
* https://www.lua.org/pil/9.1.html
