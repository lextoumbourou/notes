---
title: Step-down Rule
date: 2022-06-25 00:00
tags:
  - SoftwareEngineering
cover: /_media/story-focused-standups-cover.png
summary: Start the day by focusing on the work.
status: draft
---

A simple thing that helped us reduce decisions and made our code more readable was to add the Step-down Rule to our code style guide.

The step-down rule comes from this sentence in Clean Code:

"We want the code to read like a top-down narrative. We want every function to be followed by those at the next level of abstraction so that we can read the program, descending one level of abstraction at a time as we read down the list of functions. I call this the step-down rule."

Suppose we are writing a class-like thing. Please ensure we order it by the constructor, followed by the key behaviour it exposes, followed by everything else. It makes it much faster for a newcomer to some code to understand its purpose at a glance.

For example, for the Splash Roblox game, we write a lot of modules that look like this:

```lua
local MyThing = newMetaTable()

function MyThing:init(deps)
end

function MyThing:startKeyBehaviour()
end

function MyThing:endKeyBehaviour()
end

return MyThing
```

Sometimes as we add functionality, we find private methods scattered throughout the module.

```lua
function MyThing:_someSubBehaviour()
end

function MyThing:_anotherSubBehaviour()
end

function MyThing:startKeyBehaviour()
end

function MyThing:endKeyBehaviour()
end
```

Eventually, you add enough for these out of misplaced methods, obscuring the code's fundamental behavior. Now the reader has to do detective work to see how we interact with it elsewhere in the code base.

I think we follow this rule intuitively anyway, but in the chaos of software development, it's nice to have a reminder.