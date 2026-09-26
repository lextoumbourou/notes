---
title: "ROUGE: A Package for Automatic Evaluation of Summaries"
date: 2026-07-25 00:00
modified: 2026-07-25 00:00
status: draft
tags:
- NLPMetrics
---

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) is a family of lexical metrics that scores a candidate summary by how much it overlaps with one or more human reference summaries [@linRougePackageAutomatic2004]. Because it only measures surface word overlap, it captures content coverage but ignores meaning, factuality, and fluency, so a summary can score well while still reading poorly or misrepresenting the source.

## How the variants differ

The variants differ in *what unit of overlap* they count:

- **ROUGE-N** counts n-gram overlap. [ROUGE-1](../../permanent/rouge-1.md) is unigram (single-word) overlap and **ROUGE-2** is bigram overlap; ROUGE-1 rewards getting the right words, ROUGE-2 adds sensitivity to local word order.
- **ROUGE-L** counts the longest common subsequence, so it rewards words appearing in the same relative order without requiring them to be contiguous.
- **ROUGE-W** is a weighted LCS that favours longer *consecutive* matches over scattered ones.
- **ROUGE-S** counts skip-bigrams (word pairs in order, allowing gaps); **ROUGE-SU** adds unigrams so a candidate with no in-order pairs is not scored zero.

Each reports recall, precision, and an F-measure. ROUGE-1 is the most common baseline because it is simplest and correlates reasonably with human judgment, but it is also the easiest to game with the right bag of words.
