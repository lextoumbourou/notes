---
title: "Heavy Thinking: A Two-Stage Test-Time Scaling Pattern"
date: 2026-05-17 00:00
modified: 2026-05-17 00:00
tags:
- AgenticReasoning
- TestTimeScaling
- AgentSkills
category: reference/papers
paper_title: "HEAVYSKILL: Heavy Thinking as the Inner Skill in Agentic Harness"
paper_link: https://arxiv.org/abs/2605.02396
paper_authors: "Jianing Wang, Linsen Guo, Zhengyu Chen, Qi Guo, Hongyu Zang, Wenjie Shi, Haoxiang Ma, Xiangyu Xi, Xiaoyu Li, Wei Wang, Xunliang Cai"
---

## Overview

This paper is an empirical investigation of an [Agentic Reasoning](../../permanent/agentic-reasoning.md) approach they call [Heavy Thinking](../../permanent/heavy-thinking.md).

It's a simple two-stage workflow: a bunch of subagents reason independently in parallel on a problem, then another LLM summarises their responses.

In **Stage 1**, the step they call **Parallel Reasoning**, sub-agents attempt to solve a problem independently, encouraged to approach it from different perspectives, and output both a reasoning trace and an answer.

In **Stage 2**, which they call **Sequential Deliberation**, another LLM analyses the set of outputs from the previous step and performs meta-analysis to derive a final answer. This can be an iterative process in which the LLM deliberates across multiple steps and optionally includes the deliberation outputs in the parallel reasoning traces at each iteration (they call this **Iterative Deliberation**).

---

Since we can scale the number of parallel agents, or the "width" of reasoning, and the number of sequential deliberation steps, the "depth" of reasoning, it serves as a powerful [Test-Time Scaling](../../permanent/test-time-scaling.md) technique.

They argue that the complex orchestration systems that power tools like Codex, Claude Code, OpenClaw, and Hermes can be abstracted into this two-stage pattern.

> "we abstract the agentic harness into the LLM’s inherent capability of heavy thinking."

They also propose a memory cache mechanism to store and organise reasoning trajectories, with a few details I go into in the next section.

Finally, they propose **HeavySkill**, where they consolidate their insights into a single [Agent Skill](../../permanent/agent-skill.md).

![heavyskill-figure-1.png](_media/heavyskill/figure-1.png)

We don't know for sure, but others have speculated that GPT-5 Pro uses a similar internal pattern of parallel reasoning chains [@nateGPT5ProFirst2025], hence it is about 5x more expensive than GPT-5.

Also, reminds me a lot of the [Stacking](../../permanent/stacking.md) ensemble approach in classification-based ML, where a meta-model combines outputs from other models to improve overall performance.

---
### Serialised Memory Cache

The memory cache is the bridge between the parallel reasoning step and the deliberation step.

Since each parallel trajectory contains a reasoning trace and the final answer, the model cannot always fit every full reasoning trace into context. So the paper prunes the trajectories before passing them into the deliberation stage.

The trajectories are also shuffled. This is to stop the deliberation model from developing a position bias, where it trusts answers more just because they appear first or last in the prompt.

## HeavySkill

The workflow can be run as a concrete Python pipeline, but the authors also package it as a readable skill file they call HeavySkill, because modern agentic harnesses often support skills as plain-text instructions loaded into the model context.

The whole [Skill can be read in GitHub](https://github.com/wjn1996/HeavySkill/blob/main/skill/heavyskill.md) - it's quite readable.

## Results

They test both closed-weight (`GPT-5` with reasoning, `Claude 4.5 Thinking` and `Gemini 3 Pro Preview`) and open-weight models (`R1-Distill-Qwen-*`, `Qwen3-*`, `DeepSeek R1-0528`, `GPT-OSS-20B`, `Kimi K2 Thinking`, `GLM 4.6` and `DeepSeek V3.2 Thinking`).

By default, the same model is used for both stages, although in theory different models could be used to further improve performance.

The main result is that HeavySkill performs best on tasks with a single correct answer.

On STEM benchmarks, accuracy after deliberation (they call this **Heavy-Mean@K**) consistently outperforms simply averaging results (`Mean@K`) across models. In other words, the deliberation step usually improves upon the average single-reasoning trajectory.

Heavy thinking also often beats majority voting (which they call **Vote@K**). Even in the case where only a few of the models found the right answer, the deliberation step could reason through the correct answer, despite the fact that the majority got it wrong.

Even more surprisingly, they report that stronger models can approach the upper bound, when only one model gets the right answer (**Pass@K**). In other words, if just one of the parallel attempts found the right answer, a strong deliberation model would identify it.

The strongest results appear on difficult reasoning benchmarks like: AIME25, BeyondAIME, HMMT25-Feb, GPQA-Diamond. On harder benchmarks, the advantage over voting becomes more pronounced.

The results are more mixed outside STEM. Heavy thinking helps on correctness-oriented tasks like: LiveCodeBench, IFEval and IMO-style answer benchmarks. These tasks still have relatively clear success criteria. Code either passes tests or it does not. Instruction following can be checked. Math answers can be verified.

But the gains are weaker on Arena-Hard. Arena-Hard is more subjective and preference-based. There may not be a single correct answer. In that setting, combining multiple answers does not necessarily produce something the judge prefers.

## Further Analysis

The paper also investigates why heavy thinking works, which is probably the most significant contribution.

### Sequential deliberation can rescue low-frequency correct answers

A key finding is that deliberation can sometimes recover correct answers even when they are not the majority. If 3 out of 16 trajectories are correct, majority voting may still fail. But a deliberation model can inspect the reasoning and decide that the minority answer is actually better.

The authors describe the deliberation model as acting like an implicit verifier. It compares trajectories, identifies inconsistencies, and seeks the strongest reasoning path.

### The deliberation model does not need to be the best reasoning model

The paper also tests different models in the second-stage deliberation role. The interesting result is that the deliberation model does not necessarily need to be the strongest raw problem solver. It mainly needs to be good at reading reasoning traces, comparing arguments, identifying mistakes and synthesising a final answer

That suggests the two stages may need different capabilities. The first-stage model needs to generate diverse, high-quality attempts. The second-stage model needs to judge and summarise those attempts well. Again, this feels a lot like meta-models in [Stacking](../../permanent/stacking.md) solutions, where ensemble diversity tends to matter more than the strength of individual models.

### Iteration has a trade-off

Iterative deliberation can improve Heavy-Mean performance as more rounds of deliberation are added, but it also has a downside. The paper finds that Heavy-Pass can degrade with increasing iterations. The likely reason is that later deliberation rounds can inherit mistakes, noise, or bias from earlier summaries. So, more deliberation is not automatically better.

### Heavy thinking also works with tool use

The authors also test heavy thinking in tool-interleaved reasoning scenarios. In these experiments, models can use a Python interpreter during the parallel reasoning stage. Heavy thinking still beats majority voting across the tested models and datasets. This suggests the same pattern works even when the reasoning trajectories include tool feedback.