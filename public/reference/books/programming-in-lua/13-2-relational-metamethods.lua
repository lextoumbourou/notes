-- # 13.2 - Relational Metamethods
--
-- Metatables let you define meaning of relational operators with metamethods: __eq (equality), __lt (less than) and __le (less or equal)
-- It does not have a metamethod for ~= as that is simply not (a == b), nor > and >= as that is simply the inverse of less than.
--
--
Set = {}

function Set.union(a, b)
    local res = Set.new{}
    for k in pairs(a) do res[k] = true end
    for k in pairs(b) do res[k] = true end
    return res
end

function Set.intersection(a, b)
    local res = Set.new{}
    for k in pairs(a) do
        res[k] = b[k]
    end

    return res
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

function Set.print(s)
    print(Set.tostring(s))
end

Set.mt = {}  -- metatable for sets.

function Set.new(t)
    local set = {}
    setmetatable(set, Set.mt)
    for _, l in ipairs(t) do set[l] = true end
    return set
end

-- Add the metamethod __add to metatable.
Set.mt.__add = Set.union
Set.mt.__mul = Set.intersection

Set.mt.__le = function(a, b)
    for k in pairs(a) do
        if not b[k] then return false end
    end
    return true
end

Set.mt.__lt = function (a,b)
    return a <= b and not (b <= a)
end

s1 = Set.new{1, 10}
s2 = Set.new{10, 9, 1}

Set.mt.__eq = function (a,b)
   return a <= b and b <= a
end

print(s1 <= s2)
print(s1 < s2)
print(s1 >= s1)
print(s1 > s1)
Set.print(s2 * s1)
Set.print(s1)
print(s1 == (s2 * s1))

-- Relational metamethods don't support mixed types, unlike arithmetic metamethods.
