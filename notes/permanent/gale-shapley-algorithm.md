---
title: Gale-Shapley Algorithm
date: 2024-02-08 00:00
modified: 2024-02-08 00:00
cover: /_media/gale-shapley-cover.png
summary: an algorithm that matches 2-equally sizes groups based on preferences.
---

The Gale-Shapley algorithm (also known as **Deferred Acceptance**) solves the **stable matching problem**, where the goal is to match members of two equally sized groups based on their preferences. All pairs should be *stable*, that is, no two pairs would prefer another compared to their assigned match.

Consider a speed dating event; for simplicity of explanation, I'll assume all participants are heterosexual.

At the event of a series of conversations, each party writes their preferences in order.

Here, the preferences are expressed as Python code.

```python
men_preferences = {
    "John": ["Sally", "Jill", "Doris"],
    "Jacob": ["Sally", "Jill", "Doris"],
    "Bob": ["Sally", "Doris", "Jill"]
}
```

```python
womens_preferences = {
    "Sally": ["John", "Jacob", "Bob"],
    "Jill": ["Jacob", "John", "Bob"],
    "Doris": ["John", "Bob", "Jacob"]
}
```

Can we match them so that no pairs prefer another match to their own? For example, Doris might prefer John, but we can see that John prefers Sally.

The Gale-Shapely algorithm orchestrates a series of *proposals*. Each man proposes to their top-choice woman. If a woman receives multiple proposals, they accept the ones highest on their preference list and reject others. If a man is rejected, he proposes his next choice.

The process continues iteratively until all men are matched with women. Gale-Shapely guarantees a stable match, where no man or woman would prefer each other to their current partners.

Step 1.

All the men propose. Since everyone chose Sally, she took her first preference, John.

Step 2.

All the remaining men propose to the next on their list. Jacob chooses Jill, and Bob decides Doris. Jacob is Jill's number one preference, so she accepts. As does Doris.

Here, the algorithm is written in Python code. It's commonly executed with a while loop that continues to find proposals until no unmatched pairs exist.

```python
def gale_shapley(men_preferences, women_preferences):
    # Initial setup
    n = len(men_preferences)
    free_men = list(men_preferences.keys())
    engaged = {}
    proposed = {man: [] for man in men_preferences}

    while free_men:
        man = free_men[0]
        man_prefs = men_preferences[man]
        woman = next(w for w in man_prefs if w not in proposed[man])
        proposed[man].append(woman)

        if woman not in engaged:
            # Woman is free
            engaged[woman] = man
            free_men.remove(man)
        else:
            # Woman is engaged, check if she prefers this new man
            current_man = engaged[woman]
            if woman_prefers(man, current_man, women_preferences[woman]):
                # Woman prefers new man
                engaged[woman] = man
                free_men.remove(man)
                free_men.append(current_man)
            # Otherwise, do nothing

    return engaged
```

In the real world, the algorithm is used in kidney exchange programs to match medical graduates to residency problems, job/employer matching, and many other places.
