-- 13.4 - Table-Access Metamethods
--
-- Metamethods can also be used to change the query and modification of absent fields in a table.
--
-- 13.4.1 - The __index Metamethod
--
-- When you access a field in a table that doesn't exist, interpretor will look for a metamethod called __index. If it doesn't exist, then the field will be returned as nil
--
-- Inheritance is the main use case for this.
--
Window = {}

-- create the prototype with default values
Window.prototype = {x=0, y=0, width=100, height=100}

-- create a metatable
Window.mt = {}

-- declare the constructor function
function Window.new(o)
    setmetatable(o, Window.mt)
    return o
end

-- define __index metamethod to get value from parent
Window.mt.__index = function(table, key)
    return Window.prototype[key]
end

-- create new window and query for absent field:
w = Window.new{x=10, y=20}
print(w.width)    --> 100

-- Since this pattern is so common, there's a shortcut: __index can also reference a table. When you do this, the __index table will be looked up if the field can't be found in the original table.
Window.mt.__index = Window.prototype

-- Now when Lua looks for metatable's __index field, it find sthe value of Window.prototype, which is a table.
w = Window.new{x=10, y=20}
print(w.width)

w = Window.new{x=10, y=20, width=200}
print(w.width) -- width has been overwritten.

-- If you need to access the table without calling the __index metamethod, you can use the `rawget(t, i)` function:
print(rawget(w, 'height'))  -- returns nil

-- 13.4.2 - The __newindex Metamethod
--
-- Similar to __index but used for table accesses.
-- When you assign a value to an absent index, interpreter looks for __newindex metamethod. If found, interpreter calls it instead of assigning directly.
-- If it's a table, interpreter does assignment in that table.
-- Use `rawset(t, k, v)` to set values to original table.

-- 13.4.3 - Tables with default values

-- You can change a table's default attribute value from nil to some other value:

function setDefault (t, d)
    local mt = {__index = function () return d end}
    setmetatable(t, mt)
end

tab = {x=10, y=20}
print(tab.x, tab.z)     --> 10   nil
setDefault(tab, 1)
print(tab.x, tab.z)     --> 10   1

-- 13.4.4 - Tracking table accesses
--
-- _index and _newindex are only relevant when key doesn't exist on the table. To catch all access to a table, you have to keep an empty proxy table.
--
local t = {hello='World'}

local _t = t

-- Proxy table.
t = {}

-- Metatable
local mt = {
    __index = function(t, k)
        print('Access to element ' .. tostring(k))
	return _t[k]
    end,
    __newindex = function(t, k, v)
        print('update of element '..tostring(k)..' to '..tostring(v))
	_t[k] = v
    end
}
setmetatable(t, mt)
t[2] = 'hello'
print(t[2])

-- You cannot traverse the table with this approach - pairs function operates on the proxy not the original.

-- Monitoring several tables can be done with different metatables for each one.
local index = {}
local mt = {
    __index = function(t, k)
        print('Access to element '..tostring(k))
        return t[index][k]
    end,
    __newindex = function(t, k, v)
        print('Update of element '..tostring(k)..' to '..tostring(v))
	t[index][k] = v
    end
}
function track(t)
    local proxy = {}
    proxy[index] = t
    setmetatable(proxy, mt)
    return proxy
end

local blah = {hello='Word'}
blah = track(blah)
blah[2] = 'Hello'

-- 13.4.5 - Read-only tables
--
-- Can simply point __newindex to a function that will raise an error and __index to the original table. Note in the below example proxy stays empty in order for __newindex to do its thing.

function readOnly(t)
	local proxy = {}
	local mt = {
		__index = t,
		__newindex = function(t, k, v)
			error("attempt to update a read-only table", 2)
		end
	}
	setmetatable(proxy, mt)
	return proxy
end

days = readOnly{"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"}
print(days[1])
days[1] = "Blah"
