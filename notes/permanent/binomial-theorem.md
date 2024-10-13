---
title: Binomial Theorem
date: 2024-03-10 00:00
modified: 2024-03-10 00:00
status: draft
---

The [Binomial](binomial.md) Theorem in algebra is as follows:

$(a + b)^n = \sum\limits_{k=0}^{n} {n \choose k} a^{n-k} b^{k}$

Note that ${n \choose k}$ comes from [Combinatorics](combinatorics.md) where ${n \choose k} = \frac{n!}{k!(n - k)!}$

Examples:

$(a + b)^{4}$
$= \sum\limits_{k=0}^{4} {4 \choose k} a^{4 - k} b^{k}$
$= {4 \choose 0} a^{4} + {4 \choose 1} a^{3}b^{1} + {4 \choose 2} a^{2}b^{2} + {4 \choose 3}a^{1}b^{3} + {4 \choose 4}b^4$
$=a^4 + 4a^3b + 6a^2b^2 + 4ab^3 + b^4$

$(a + b)^{3}$
$= \sum\limits_{k=0}^{3} {3 \choose k} a^{3 - k} b^{k}$
$= {3 \choose 0} a^{3} + {3 \choose 1} a^{2}b + {3 \choose 2} a^{1}b^{2} + {3 \choose 3}b^{3}$
$=a^3 + 3a^2b + 4a^{1}b^2 + b^3$
