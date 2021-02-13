-- # 13 – Metatables and Metamethods
--
-- Metatables allow for changing behaviour of table
--   If you try to add 2 tables with +, Lua checks whether either values has a metatable with an __add field.
--
--   Each table can have own metatable. New tables are always created without metatables
t = {}
print(getmetatable(t))

-- Can use `setmetatable(t, t1)` to set metatable.
t1 = {}
setmetatable(t, t1)
assert(getmetatable(t) == t1)

-- Any table can be a metatable of another table; group of table can share common metatable and table can be own metatable.
