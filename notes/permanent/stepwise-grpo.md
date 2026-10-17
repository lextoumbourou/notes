---
title: "Stepwise GRPO"
date: 2026-10-18 08:35
modified: 2026-10-18 09:09
status: hidden
summary: "Group Relative Policy Optimisation applied to multi-turn agent trajectories, training each model call against the context it actually saw."
tags:
- ReasoningModels
- AgenticReasoning
---

[Group Relative Policy Optimisation](group-relative-policy-optimisation.md) (GRPO) applied to multi-turn agent trajectories, training each model call against the context it actually saw.

GRPO comes from the DeepSeekMath paper [@shaoDeepSeekMathPushingLimits2024]. For each question, you sample a group of outputs, score them, and normalise each reward against the group's mean and standard deviation. That normalised reward is the advantage, so no value model is needed. With outcome supervision, every token in an output gets the same advantage.

In an agent trajectory, the output is a sequence of model calls, each with its own input. If the context changes between calls (because of summarisation or edits), the trajectory can't be trained as one long sequence. Stepwise GRPO splits the trajectory into steps, keeps each step with its original input, and gives every step the trajectory's group-normalised advantage.

DeepSeekMath doesn't use the name "stepwise GRPO". Its closest variant is process supervision, where a reward model scores each reasoning step and a token's advantage is the sum of the normalised rewards for the steps from there to the end. [Context Language Models](context-language-models.md) cites it for stepwise GRPO, but uses the simpler outcome version, broadcasting the final advantage to every step.
