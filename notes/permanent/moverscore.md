---
title: MoverScore
date: 2026-07-25 00:00
modified: 2026-07-25 00:00
status: hidden
source: https://arxiv.org/abs/1909.02622
tags:
- NaturalLanguageProcessing
- Metrics
---

**MoverScore** scores a candidate against a reference over contextual embeddings, but instead of [BERTScore](bertscore.md)'s greedy one-to-one token matching it finds the minimal Earth Mover's Distance (Word Mover's Distance) needed to transform one text's embeddings into the other's [@zhaoMoverScoreTextGeneration2019]. This soft, many-to-one alignment lets partial and distributed matches count, which tends to track human judgment better than hard matching, though it is still a single opaque similarity number that says nothing about factuality.
