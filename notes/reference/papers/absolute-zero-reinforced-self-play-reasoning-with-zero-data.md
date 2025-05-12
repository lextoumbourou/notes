---
title: "Absolute Zero: Reinforced Self-play Reasoning with Zero Data"
date: 2025-05-12 00:00
modified: 2025-05-12 00:00
summary: learn to reason without any human-annotated data.
cover: "/_media/absolute-zero.png"
hide_cover_in_article: true
tags:
- ReinforcementLearning
- ReasoningModels
- LargeLanguageModels
---

*My summary of [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://www.arxiv.org/abs/2505.03335) by Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang.*

This paper introduces the [Absolute Zero Reasoner](../../permanent/absolute-zero-reasoner.md).

The idea is to train a reasoning model (from a foundation Qwen-2.5-7B* model) without needing any human-annotated data. Previous [DeepSeek-R1-Zero](../../permanent/DeepSeek-R1-Zero.md) approach learned CoT reasoning just from input/output pairs, but this goes a step further and proposes the inputs and outputs to solve.

![absolute-zero-comparison.png](_media/absolute-zero-comparison.png)

The proposing and solving steps both have relevant reward functions: they reward the model to propose examples that are the correct level of difficulty for it, and penalise wrong answers in the solving step.

The tasks it proposes and solves are either:

- **deduction** (prediction output, given program and input)
- **abduction** (prediction input, given program and output)
- **induction** (predict/synthesis program, given input and output)

![propose-solve.png](_media/propose-solve.png)


Of note, is that they also notice some potentially unsafe reasoning chains generated throughout the training, which they call an "uh-oh moment".

![uh-oh-moment.png](_media/uh-oh-moment.png)

It's pretty wild that this works. In theory, we can improve reasoning capability of models by just training them longer with no additional data needed, although I'm a bit dubious about how far it can be pushed, given that the proposing step is still limited to the distribution of the foundation model's training data.