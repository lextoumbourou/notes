---
title: Cross-Entropy
date: 2021-07-29 00:00
tags:
  - InformationTheory
cover: /_media/cross-entropy-cover.png
---

Cross-entropy measures the average number of bits required to identify an event if you had a coding scheme optimised for one probability distribution $q$, where the true probability distribution was actually $p$.

It's the same as [Information Entropy (Information Theory)](../reference/moocs/khan-academy/information-theory/information-entropy.md) but measuring what happens if you have are identifying messages using a different probability distribution.

Expressed as: $$H(p, q)=-\sum\limits_{i=1}^{n} p_{i} \times log_2(q_{i})$$

[@CrossEntropy2021]
