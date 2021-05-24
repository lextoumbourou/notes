-- # 13.3 - Library-Defined Metamethods
--
-- Lua core is not the only place that makes use of metamethods: libraries can also use them.
-- One example is the `tostring` method. It looks for metamethod `__tostring`.
--

Set = {}
Set.mt = {}  -- metatable for sets.
function Set.new(t)
    local set = {}
    setmetatable(set, Set.mt)
    for _, l in ipairs(t) do set[l] = true end
    return set
end

function Set.tostring(set)
    local s = "{"
    local sep = ""
    for e in pairs(set) do
        s = s .. sep .. e
	sep = ", "
    end

    return s .. "}"
end

Set.mt.__tostring = Set.tostring

-- Now we can use print method on set (which calls tostring behind the scenes)

s1 = Set.new{10, 4, 5}
print(s1)

-- Even setmetatable/getmetatable use metafields too, to be able to protect metatables.
-- If you set __metatable field on a metatable, that will be returned when getmetatable is called and it will not be settable.
