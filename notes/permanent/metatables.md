---
title: Metatables
date: 2021-02-14 00:00
tags:
  - Lua
summary: A table attached to another table to define its custom behaviour, via metamethods like __add and __index.
---

In Lua, since everything is a table, metatables are a paradigm that lets you change the behaviour of a table, using another table. For example, you can define the behaviour when 2 tables are added together using the `+` operator by defining a metatable with an `__add` method. This is similar to the concept of [magic methods (Python)](magic methods (Python)), though in Python magic methods are defined on instance's class.

In this example, I'm defining addition as the sum of all keys in the left-most table

```lua
local metatable = {
    __add = function(self, other)
        local output = {}
        for k, v in pairs(self) do
            output[k] = self[k] + other[k]
        end
        return output
    end
}

local t = {A=100, B=200}

setmetatable(t, metatable)

local result = t + {A=50, B=100}

print(result['A'])  -- 150
print(result['B'])  -- 300
```

`__add` is an arithmetic metamethod, alongside `__mul` (multiplication behaviour of `*` between 2 tables), `__div` (division), `__sub` (subtraction), `__unm` (negation) and `__pow` (exponential).

There are also relational metamethods for comparison: `__eq`, `__lt` and `__le`

Lastly, [Lua Table-Access Metamethods](lua-table-access-metamethods.md) allow for defining behaviour when missing keys are looked up.

References:

* [Programming in Lua - Chapter 13 - Metatables and Metamethods](https://www.lua.org/pil/13.html)
