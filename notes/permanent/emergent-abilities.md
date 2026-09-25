---
title: Emergent Abilities
date: 2026-09-25 17:35
modified: 2026-09-25 20:19
status: draft
tags:
- LargeLanguageModels
aliases:
- Emergent Ability
- Emergent Property
---

**Emergent Abilities**, in the context of [Large Language Models](large-language-models.md), are abilities that are "not present in smaller models but [are] present in larger models" [@weiEmergentAbilitiesLarge2022]. Performance stays near random as models get bigger, then jumps once they pass some scale. You can't predict it by extrapolating from the smaller models.

The term comes from the idea of [Emergence](emergence.md) more generally. Wei et al. root it in physicist Philip Anderson's 1972 essay "More Is Different" [@andersonMoreDifferent1972], summarising it as: "Emergence is when quantitative changes in a system result in qualitative changes in behavior". In LLMs, the quantity is scale: training compute and parameter count.

[Chain-of-Thought Prompting](chain-of-thought-prompting.md) is one of their main examples. It only beats standard prompting at around 100B parameters [@weiChainofThoughtPromptingElicits2022]. Scratchpads and [Self-Consistency](../reference/papers/self-consistency-improves-chain-of-thought-reasoning-in-language-models.md) decoding are also on their list.

However, not everyone agrees that these jumps are real. Schaeffer et al. argue that emergent abilities "appear due to the researcher's choice of metric rather than due to fundamental changes in model behavior with scale" [@schaefferAreEmergentAbilities2023]. All-or-nothing metrics like exact match make performance look like it jumps suddenly. Measure the same models with a smoother metric and the improvement is gradual and predictable.
