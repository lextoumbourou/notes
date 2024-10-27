---
title: "Statistical Phrase-Based Translation (2003)"
date: 2024-10-28 00:00
modified: 2024-10-28 00:00
status: draft
---

My notes from paper [Statistical Phrase-Based Translation](https://aclanthology.org/N03-1017.pdf) (2013) by Philipp Koehn, Franz Josef Och, Daniel Marcu.

## Overview

This paper presents a statistical phrase-based translation model and decoding algorithm for machine translation. The researchers evaluate and compare various methods for extracting phrase translations from parallel corpora, including word-based alignments and syntactic parsing. They find that a relatively simple approach, using heuristic learning of phrase translations from word-based alignments and lexical weighting, achieves high levels of performance. Surprisingly, learning phrases longer than three words and using more complex alignment models does not significantly improve performance. Furthermore, restricting phrase pairs to only those with a syntactic motivation actually degrades performance. The paper concludes that phrase-based translation outperforms traditional word-based methods and provides insights into the best practices for building effective phrase translation models.
