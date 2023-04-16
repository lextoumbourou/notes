---
title: Object Prototypes
date: 2021-02-15 00:00
tags:
  - Lua
---

In Lua, object-oriented programming is achieved using prototypal inheritance, since Lua does not have the concept of a [class](class). Prototypal inheritance simply means that objects can refer to a "prototype" object which is looked up when any field doesn't exist in the first object. In Lua, this is achieved by using the `__index` [Lua Table-Access Metamethods](lua-table-access-metamethods.md). In the example, I'm creating a `Person` table which will serve as a metatable for all instances of a `Person`:

```lua
local Person = {firstName='', lastName='', age=nil}
function Person:new(o)
    local o = o or {}
    setmetatable(self, o)
    self.__index = self
    return o
end

function Person:getName()
    return self.firstName..' '..self.lastName
end

local me = Person({firstName='Lex', lastName='T', age=34})
print(me:getName())
```

I can then extend Person by creating a new object with modified parameters:

```lua
local PersonWithMiddleName = Person:new()
PersonWithMiddleName.middleName = ''

local me = PersonWithMiddleName(firstName='Lex', lastName='T', age=34, middleName='D')
```

In that example, the `me` object will be consulted for fields. If they aren't found, `PersonWithMiddleName` will be looked up. If not found, since it uses `Person` as the `__index` metatable, `Person` will then be consulted.

Javascript also uses prototypal inheritance at the core of its object-oriented paradigm.

References:

* [Programming in Lua, Chapter 16](https://www.lua.org/pil/16.html)
