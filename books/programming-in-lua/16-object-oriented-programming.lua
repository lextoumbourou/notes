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
