---
title: Workflow for Really Difficult Bugs
date: 2022-07-27 00:00
---

When multiple customers are reporting a bug that you, or anyone on your team can't replicate, what do you do?

* Summaries the issue, and include screenshots of customer reports if possible.
* Include your initial hypothesises about the issue:
    * New accessories or stage props have high triangle counts and are causing lag.
    * The Kai fireworks are laggy.
        * Anecdotal evidence that a server I was on seemed to go really slow when Kai’s fireworks started.
    * Some other event is laggy or has a memory leak.
* Note down each thing you have investigated.

## Summary



I like to create a document that describes the bug and includes all the customer reports.

Sometimes you get halfway through this process and figure out what the bug is and move on with your life.

That's great.

What is the bug?



Once we can replicate our bug, it's only a matter of time before we fix it.

But the bugs we can't replicate, or the ones we seldom replicate, 

1. What do the reports have in common? What differentiates them?
2. How can I collect more information.
3. Perhaps I suspect a race condition? Can I inject pauses?

---

