-- 16 - Object-Oriented Programming
--
-- Tables in Lua are objects: they have state, identity and can have operations (methods).
--
-- To create a method in Lua, you would create a function on a table and pass `self` as the first argument:
local User = {firstName='John', lastName='Smith'}
function User.getFullName(self)
	return self.firstName..' '..self.lastName
end

local u = User
print(u.getFullName(u))

-- Lua provides a syntactic facilities to hide the `self` param using the colon syntax:
print(u:getFullName())

-- 16.1 Classes
--
-- In other OO languages with a concept of class, a class is used to create instances of that class.
-- Lua doesn't have the concept of class, but it can be emualated using an object prototype (similar to JavaScript)
-- A prototype is simply a regular object which is looked up when the first object doesn't know about a field.
--
-- It's trivial to create a prototype object using metatables:
a = {hello=1}
b = {world=2}
setmetatable(a, {__index=b})
print(a.world)  -- 2

-- You can then define a constructor that creates a new object, and uses the main table (aka emulated class) as the prototype.
local Person = {firstName='None', lastName='None'}

function Person:getFullName()
    return self.firstName..' '..self.lastName
end

function Person:new(o)
    local o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

local me = Person:new({firstName='Lex', lastName='T'})
print(me:getFullName())

-- 16.2 Inheritance
--
-- You can simple create a new object that inherits from the first.
local PersonWithMiddleName = Person:new({})

PersonWithMiddleName.middleName = ''

function PersonWithMiddleName:getFullName()
    return self.firstName..' '..self.middleName..' '..self.lastName
end

local meWithMiddleName = PersonWithMiddleName:new({firstName='Lex', middleName='D', lastName='T'})
print(meWithMiddleName:getFullName())

-- 16.3 - Multiple interitance
--
-- Multiple interitance is when a class has more than one superclass. Can be achieved in Lua by setting __index as a function and looking up multiple parents.

local function search(k, parentList)
    for i=1, #parentList do
        local v = parentList[i][k]
	if v then return v end
    end
end

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

-- This is a lower performance option: one way to improve is to copy the function into the subclass.

--[[
setmetatable(c, {__index = function (t, k)
    local v = search(k, arg)
    t[k] = v       -- save for next access
    return v
end})
]]--

-- Privacy
--
-- Goal of Lua is to be a large for building small to medium-size projects. Secondary goal is to be flexible, with meta mechanism allowing for many different programming constructs.
-- Privacy can be achieved using an internal table that is only kept in closure of methods.
function newAccount(initBalance)
    local self = {balance = initBalance}

    local withdraw = function (v)
        self.balance = self.balance - v
    end

    return {
        withdraw = withdraw,
	getBalance=function() return self.balance end
    }
end

acc1 = newAccount(100.00)
acc1.withdraw(40.00)
print(acc1.getBalance())

-- In the above example `self` table is not available in outside scope, only through methods.
--
-- Another approach to OO is to create an interface that returns a single method for modifying some internal state. Could also perform different tasks based on passed argument.
function newObject(value)
    return function(action, v)
        if action == "get" then return value
	elseif action == "set" then value = v
	else error("invalid action")
	end
    end
end

d = newObject(0)
print(d("get"))    --> 0
d("set", 10)
print(d("get"))    --> 10
