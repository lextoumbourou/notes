---
title: Order Methods Like A Newspaper Article
date: 2022-06-27 00:00
tags:
  - SoftwareEngineering
cover: /_media/ordering-methods-cover.png
summary: Order your methods so your code reads like a newspaper.
---

One simple thing that made our code more readable was to add a code style guide rule for ordering methods

In Clean Code by Robert C. Martin, he applies The Newspaper Metaphor to vertical formatting of code:

> "Think of a well-written newspaper article. You read it vertically. At the top, you see a headline that will tell you what the story is about and allow you to decide if you want to read it. The first paragraph gives you a synopsis of the whole story, which hides all the details while giving you the broad-brush concepts. As you continue downward, the details increase until you have all the dates, names, quotes, claims, and other minutia." [^1]

For our project, where Lua is the primary language, this is the snippet of our style guide about ordering methods:

> # Ordering methods
> 
> Order your methods so that the module starts with the constructor (if required), followed by the public methods it implements, then any private implementation details.

When we follow this rule consistently, we immediately know at a glance how to create an instance of our objects and can see their fundamental purpose. We don't have to hunt around to see how we interact with it elsewhere in the code base to get a sense of it.

Some modules don't need constructors; some just expose read-only variables. We can spot those quickly, too, as they don't start with a constructor.

Here's an example of following this rule in a UI module whose job is to render something on the screen.

```lua
local MyThing = newComponent()

-- Show me how to create one of these.
function MyThing:init(deps)
end

-- The key behavior it implements.
function MyThing:render()
end

-- All the implementation details below.
function MyThing:_implementationDetail1()
end

function MyThing:_implementationDetail2()
end

return MyThing
```

I think we intuitively follow this rule anyway, but it is easy to overlook.

Cover by [Rishabh Sharma](https://unsplash.com/@rishabhben) on Unsplash.

[^1]: MLA (7th ed.) Martin, Robert C. Clean Code: A Handbook of Agile Software Craftsmanship. Upper Saddle River, NJ: Prentice Hall, 2009. (Pg. 77)