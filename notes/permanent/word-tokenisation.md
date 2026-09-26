---
title: Word Tokenisation
date: 2025-11-03 00:00
modified: 2026-09-26 08:55
status: draft
---

**Word Tokenisation** is (obviously) a type of [Tokenisation](tokenisation.md) concerned with converting text into word-level tokens.

For example, `I don't like cats.` could be split into `["I", "don't", "like", "cats", "."]`.

Splitting on whitespace is the simplest approach, but it has to deal with punctuation and contractions, and it can't represent words that weren't seen during training (out-of-vocabulary words). That's one reason modern language models use subword tokenisation instead.
