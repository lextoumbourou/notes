---
title: Entropy (Information Theory)
date: 2021-07-28 20:45
tags:
  - InformationTheory
cover: /_media/claude-shannon-entropy.jpeg
---

Entropy is a measure of uncertainty of a variable's possible outcomes.

It's highest when all outcomes are equally likely. As you introduce more predictability (one of the variables has a higher probability), Entropy decreases.

It measures many "questions" on average you need to guess a value from the distribution. Since you'd start by asking the question that is most likely to get the correct answer, distributions with low Entropy would require less bandwidth on average to send.

The unit of Entropy is a bit.

The entropy of a distribution is as follows: $$H=\sum\limits_{n}^{i=1} p_{i} \times log_2(\frac{1}{p})$$ o r, the negative exponent is inverted and it's rewritten like this: $$H=-\sum\limits_{n}^{i=1} p_{i} \times log_2(p_i)$$

In code:

{% notebook permanent/notebooks/entropy.ipynb %}

Claude Shannon borrowed the term [[Entropy]] from thermodynamics as part of his theory of communication.

[@khanacademylabsInformationEntropyJourney]

Cover from [How Claude Shannon Invented the Future](https://www.quantamagazine.org/how-claude-shannons-information-theory-invented-the-future-20201222/).