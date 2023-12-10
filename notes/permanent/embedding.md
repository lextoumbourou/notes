---
title: Embedding
date: 2022-03-04 00:00
status: draft
---

A vector of numbers that provides a representation or, effectively a summary something.

Imagine writing an algorithm to recommend movies.

One approach is to create a [Vector](vector.md) that describes "features" of a movie. For example, it is romantic, action, arthouse etc. Then ask customers to describe how much they like each feature, and then find the movie that best matches.

However, that's not only tedious to do, it's also very hard to pick the optimal collections of features that covers all movies.

What about instead if you could give a computer room to find the best features to descrive movies based on trying to predict user's movie ratings.
