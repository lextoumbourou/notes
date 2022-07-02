---
title: An Interface Should Respond Within 0.1 Seconds
date: 2022-07-02 00:00
tags:
  - UserInterface
cover: /_media/loading-state-cover.png
summary: Developers should be vigilant of slow returning user interactions.
status: draft
---

One area where developers can significantly impact the design of an interface is to highlight interactions that will likely be slow to respond.

In The Art of Game Design, Schell says that:

> "if your interface does not respond within a tenth of a second, the player will feel like something is wrong with the interface." [^1]

A study on [Tolerable Wait Times](http://aisel.aisnet.org/amcis2003/285) by Fiona Nah[^2] concurs with the 1/10 of a second rule. 

For development on the Splash game, we've simplified it to these design rules of thumb:

1. any button press that triggers a server invocation must include a loading state.
2. any button press that requires over 10 seconds of wait time should be asynchronous, allowing the user to do other things while they wait.

It's easy to overlook this. When you're developing locally, a call to the server only has to traverse your local network and may seem responsive enough. Even in staging, with so little competition from other users, calls to your servers may be fast enough that you don't notice interface blocking.

But in production, once your users have requests that need to traverse underground cable networks or talk to slow DNS servers or [any number of things between their keyboard and a server](https://github.com/alex/what-happens-when), they will inevitably find the interface laggy, unresponsive and frustrating. 

As developers, only we know which interface interactions can return results straight from the client and which need to fetch results from servers. Only we know which requests can be produced quickly from a cache and which will require expensive processing.

We must highlight these interactions and ensure our designs have adequate means of providing immediate feedback, like loading states and progress indicators.

These rules of thumb are less critical for those building server-rendered websites as the browser elegantly handles feedback about loading states as the user clicks through links.

See also [[Goal Of A Game Interface]].

[^1]: Schell, Jesse. The Art of Game Design : a Book of Lenses. Amsterdam ; Boston :Elsevier/Morgan Kaufmann, 2008.
[^2]: Nah, Fiona, "A Study on Tolerable Waiting Time: How Long Are Web Users Willing to Wait?" (2003). AMCIS 2003 Proceedings. 285.
http://aisel.aisnet.org/amcis2003/285

Photo by <a href="https://unsplash.com/@mike_van_den_bos?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Mike van den Bos</a> on <a href="https://unsplash.com/s/photos/loading?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>