---
title: Gale-Shapley Algorithm
date: 2024-02-08 00:00
modified: 2024-02-08 00:00
cover: /_media/gale-shapley-cover.png
summary: an algorithm that matches 2-equally sizes groups based on preferences.
---

The Gale-Shapley algorithm (also known as **Deferred Acceptance**) solves the **stable matching problem**, where the goal is to match members of 2 equally sized groups based on their preferences while avoiding unstable pairs.

It works when you have one group and propose to another group of members based on a hierarchy of preferences. Proposals are either accepted or rejected according to the preference rules of the receiving group.

Imagine you have ten musicians and ten bands looking for musicians. Each band ranks musicians according to their preferences, which could be based on criteria like preferred instrument, skill level, experience, etc. Conversely, each musician ranks the bands based on genre, vibe or practice location.

The Gale-Shapely algorithm then orchestrates a series of *proposals*. Each band initially proposes to their top-choice musician. If a musician receives multiple proposals, they will tentatively accept the one from the band highest on their preference list and reject others. Bands whose proposals are rejected then propose to their next choice.

The process continues iteratively, with rejected bands making proposals to their next -preference musicians and musicians reconsidering their options until even the musician has a band.

The gale-Shapely algorithm guarantees a stable match, where no band and musician would prefer each other to their current partners.

The algorithm is used in the real world for matching medical graduates to residency problems in the US, in kidney exchange programs, for matching in job markets and many other places.

In conclusion, the Gale-Shapley algorithm efficiently resolves the stable matching problem by ensuring that all participants end up in stable pairs, meaning no two individuals would prefer to be matched over their current partners.