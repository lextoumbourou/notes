---
title: Stemming
date: 2024-07-28 00:00
modified: 2024-07-28 00:00
status: draft
---

**Stemming** is a text normalisation technique in [Natural Language Processing](../../../permanent/natural-language-processing.md) that is used to reduce a word to its common root form. It uses a crude, rule-based approach which typically chops of common suffixes like `-ing`, `-eg`, `ly`. It is fast, but unlike [Lemmatisation](../../../permanent/lemmatisation.md) it produces non-words: `argue` -> `argu`.