---
title: The Stepdown Rule
date: 2022-06-27 00:00
tags:
  - SoftwareEngineering
cover: /_media/the-stepdown-rule-cover.png
summary: A rule for ordering methods
---

One simple thing that made our code more readable was to add a code style guide rule for ordering methods.

> Order your methods so that the module starts with the constructor (if required), followed by the key public methods, then any private implementation details.

We based this rule on The Stepdown Rule from Clean Code in the chapter on functions [^1]:

> "We want the code to read like a top-down narrative. We want every function to be followed by those at the next level of abstraction so that we can read the program descending one level of abstraction at a time as we read down the list of functions."

Most intuitively follow this anyway, but it is easy to overlook.

Public methods first, preferably only a few, ensure the module's key functionality is immediately apparent.

The exact order you prefer matters less than being consistent. The cost of searching through arbitrarily ordered modules accumulates over time.

Below is an example of a typical module we'd write in Lua ordered with our style guide rule in mind.

*Note that Lua uses [[Metatables]] modules to emulate classes using [[Object Prototypes]] like Javascript. The details are unimportant, included for completeness.*

```lua
local UsefulThing = {}
-- Lua Metatable thing.
UsefulThing.__index = UsefulThing

-- Constructor
function UsefulThing.new()
    local self = {}
    -- Another Lua Metatable thing.
    setmetatable(UsefulThing, self)
    return self
end

-- The key public method
function UsefulThing:use()
    self:_prepare()
    self:_doBehaviour()
    self:_cleanUp()
end

-- All the implementation details are below
function UsefulThing:_prepare()
end

function UsefulThing:_doBehaviour()
end

function UsefulThing:_cleanUp()
end

return UsefulThing
```

Like the Python convention, we prefix private methods with underscores to further distinguish.


Photo by <a href="https://unsplash.com/@andrew23brandy?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Andrew Brandy</a> on <a href="https://unsplash.com/s/photos/complexity-step?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
[^1]: MLA (7th ed.) Martin, Robert C. Clean Code: A Handbook of Agile Software Craftsmanship. Upper Saddle River, NJ: Prentice Hall, 2009. (Pg. 37)