-- # 13.1 – Arithmetic Metamethods
--
-- Set example:
--
Set = {}

function Set.new(t)
    local set = {}
    for _, l in ipairs(t) do set[l] = true end
    return set
end

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

s1 = Set.new{10, 20, 30, 50}
s2 = Set.new{30, 1}
s3 = s1 + s2
Set.print(s3)  --> {1, 10, 20, 30, 50}

-- May use multiplicatoin operator to perform set intersection:
Set.mt.__mul = Set.intersection
Set.print((s1 + s2) * s1)

-- Can check type of operands before attempting to perform the operation:
function Set.union(a, b)
    if getmetatable(a) ~= Set.mt or
       getmetatable(b) ~= Set.mt then
       error("attempt to add a set with a non-set value", 2)
    end
    local res = Set.new{}
    for k in pairs(a) do res[k] = true end
    for k in pairs(b) do res[k] = true end
    return res
end

Set.mt.__add = Set.union

s1 = Set.new{10, 20, 30, 50}
s3 = s1 + 8 -- will raise error.
