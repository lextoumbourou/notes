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

We based this rule on The Stepdown Rule in the chapter on functions, Clean Code by Robert C. Martin [^1]:

> "We want the code to read like a top-down narrative. We want every function to be followed by those at the next level of abstraction so that we can read the program descending one level of abstraction at a time as we read down the list of functions. I call this The Stepdown Rule."

Most intuitively follow this anyway, but it is easy to overlook.

It means you can, at a glance, see how to get an instance of it and interact with it.

It might seem unimportant, but the pain of dealing with lots of unordered modules accumulates over time.

Below is an example of a typical module we'd write in Lua formatted with this rule in mind.

```lua
local UsefulThing = {}
-- Metatable implementation detail for create objects..
UsefulThing.__index = UsefulThing

-- Constructor
function UsefulThing.new()
    local self = {}
    setmetatable(UsefulThing, self)
    return self
end

-- The key public methods
function UsefulThing:use()
    self:_prepare()
    self:_doBehaviour()
    self:_cleanUp()
end

-- All the implementation details are below.
function UsefulThing:_prepare()
end

function UsefulThing:_doBehaviour()
end

function UsefulThing:_cleanUp()
end

return UsefulThing
```

Note that Lua uses [[Metatables]] to emulate classes using [[Object Prototypes]] like Javascript. The details are unimportant, included for completeness.

Photo by <a href="https://unsplash.com/@andrew23brandy?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Andrew Brandy</a> on <a href="https://unsplash.com/s/photos/complexity-step?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
[^1]: MLA (7th ed.) Martin, Robert C. Clean Code: A Handbook of Agile Software Craftsmanship. Upper Saddle River, NJ: Prentice Hall, 2009. (Pg. 37)