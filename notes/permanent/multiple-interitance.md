---
title: Multiple Interitance
date: 2021-02-21 00:00
tags:
  - Programming 
---

Multiple interitance refers to an object that inheritants from multiple parents. In Lua, this can be acheived by using [Metatables](metatables.md) with a metamethod `__index` which can search multiple parent tables for an attribute or a method.

```lua
function createClass(...)
    local arg = {...}
    local c = {}
    setmetatable(c, {__index = function (t, k)
        return search(k, arg)
    end})

    c.__index = c

    function c:new (o)
        o = o or {}
        setmetatable(o, c)
        return o
    end

    return c
end

Named = {}
function Named:getname ()
  return self.name
end

Account = {}
function Account:setname (n)
    self.name = n
end

NamedAccount = createClass(Account, Named)
NamedAccount:setname('Lex')
print(NamedAccount:getname())
```

References:

* [Programming in Lua - Chapter 16](https://www.lua.org/pil/16.3.html)
