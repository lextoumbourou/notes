Tags: #Programming 

---

# Multiple Interitance

From the book, Programming in Lua in Chapter 16, multiple interitance describes an object that inheritants from multiple parents. In Lua, this can be acheived by using [[metatables (Lua)]] with a metamethod `__index` which can search multiple parent tables for an attribute or a method.

```
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