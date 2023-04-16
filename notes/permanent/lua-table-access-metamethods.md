---
title: Lua Table-Access Metamethods
date: 2021-02-14 00:00
tags:
  - Lua
---

Table access [metamethods (Lua)](metamethods (Lua)) provide a mechanism for handling table lookup using the `__index` metamethod and writes to missing keys, using the `__newindex` metamethod. They are the foundation for object-oriented programming in Lua using [Object Prototypes](object-prototypes.md)es.md) and are also useful for creating read-only tables, tables with default values and tracking tables access.

The `__index` field in a metatable can either refer to a function that will be called each time a missing key is looked up or another table which will be looked up if the original table doesn't have the key.

In this example, I'm creating a read-only table by pointing `__index` to the table with the data we're interested in and `__newindex` to a function that will raise an error. Note that the `proxy` table has to remain empty in order for `__newindex` to be called

```lua
function readOnly(t)
    local mt = {
        __index = t ,
        __newindex = function(t, v, k)
            error('Table cannot be written to', 2)
        end
    }
    local proxy = {}  -- will remain empty
    setmetatable(proxy, mt)
    return proxy
end

local constants = readOnly({"One", "Two", "Three"})
print(constants[1])  -- return 1
constants[1] = "Two"  -- should raise read-only error
```

References:

* [Programming in Lua - Chapter 13](https://www.lua.org/pil/13.html)
