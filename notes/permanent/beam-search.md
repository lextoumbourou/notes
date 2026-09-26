---
title: Beam Search
date: 2026-09-26 08:50
modified: 2026-09-26 08:50
status: draft
---

**Beam Search** is a decoding algorithm that keeps the B most probable partial sequences at each step, instead of only the single most probable one. B is called the beam width.

At each step, every sequence in the beam is extended with each possible next token, the candidates are scored by their total (log) probability, and only the top B are kept. This continues until the kept sequences end with an end-of-sequence token.

Greedy decoding is just beam search with B=1. Larger beams usually find more probable sequences but cost more compute and memory, and because longer sequences multiply more probabilities together, scores are usually normalised by length.

See [Natural Language Processing with NLP - Week 1](../reference/moocs/coursera/attention-models-in-nlp/week-1.md).
