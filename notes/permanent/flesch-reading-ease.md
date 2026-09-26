---
title: Flesch Reading Ease
date: 2024-12-22 00:00
modified: 2026-09-26 08:55
status: draft
---

**Flesch Reading Ease** is a readability score for English text, developed by Rudolf Flesch in 1948. It's based on two things: average sentence length and average number of syllables per word.

$$
\text{FRE} = 206.835 - 1.015 \left(\frac{\text{total words}}{\text{total sentences}}\right) - 84.6 \left(\frac{\text{total syllables}}{\text{total words}}\right)
$$

Higher scores mean easier text. Scores around 60 to 70 are considered plain English, while scores below 30 are very difficult to read. The Flesch-Kincaid Grade Level is a related formula that maps the same inputs to a US school grade.

Because it only counts surface features, it can't tell whether the text actually makes sense. See [Cloze procedure: A new tool for measuring readability](cloze-procedure%20a-new-tool-for-measuring-readability.md) for an approach that tests real readers instead.
