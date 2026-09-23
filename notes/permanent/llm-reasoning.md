---
title: LLM Reasoning
date: 2025-07-28 00:00
modified: 2026-09-24 07:47
summary: How LLMs use intermediate reasoning tokens, from chain-of-thought prompting to trained reasoning models, and how this relates to other forms of reasoning.
cover: /_media/llm-reasoning-chess-knights-hassan-pasha.jpg
cover_credits: Photo by <a href="https://unsplash.com/@hpzworkz">Hassan Pasha</a> on <a href="https://unsplash.com/photos/black-and-white-chess-knights-7SjEuEF06Zw">Unsplash</a>
tags:
  - LargeLanguageModels
  - ReasoningModels
---

Reasoning, in the context of [LLMs](large-language-models.md), refers to the process of generating intermediate outputs before giving an answer. Typically, these outputs are tokens, which we call *reasoning* or *thinking* tokens. And when the model is reasoning in token-space it's also referred to as [Chain-of-Thought Reasoning](chain-of-thought-reasoning.md) [@raschkaBuildReasoningModel2026].

However, there are also approaches where the outputs are invisible to the user and the model reasons internally. One example is [Latent Reasoning](latent-reasoning.md), which performs intermediate reasoning in hidden representations rather than generating a text token for each step [@haoContinuousLatentSpace2024].

A landmark paper in the development of LLM reasoning was [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](../../../reference/chain-of-thought-prompting-elicits-reasoning-in-large-language-models.md), first released in January 2022. The researchers showed that giving the model examples with intermediate reasoning steps substantially improved its ability on a range of arithmetic, commonsense and symbolic reasoning tasks [@weiChainOfThoughtPrompting2022].

Later that year, [Large Language Models are Zero-Shot Reasoners (May 2022)](../reference/papers/large-language-models-are-zero-shot-reasoners-may-2022.md) showed that simply prompting the model with "Let's think step by step" before giving an answer could substantially improve its performance on a range of maths and logic tasks, without providing worked examples [@kojimaZeroShotReasoners2022].

Later, OpenAI trained its o1 models specifically to reason before giving an answer, using reinforcement learning. The first releases, o1-preview and o1-mini, arrived in September 2024. The o1 API release in December 2024 added a reasoning-effort control, allowing developers to influence how much reasoning the model performs [@openaiO1Model] [@openaiAPIChangelog]. See also [OpenAI o1 System Card](../../../permanent/openai-o1-system-card.md).

This helped popularise the [Test-Time Scaling](../../../permanent/test-time-scaling.md) paradigm, allowing the models to spend a configurable amount of computation when answering a question to improve the result. Though this idea predates o1 and includes approaches such as sampling multiple reasoning paths and selecting a consistent answer, as well as generating longer reasoning traces [@wangSelfConsistency2022].

In January 2025, DeepSeek released R1 and shared a training recipe, allowing us to understand in more detail how a reasoning model can be trained. Their paper, [DeepSeek-R1 Reasoning via Reinforcement Learning](../../../permanent/deepseek-r1-reasoning-via-reinforcement-learning.md), distinguishes [DeepSeek-R1-Zero](DeepSeek-R1-Zero.md), which applied reinforcement learning directly to a pretrained base model, from R1, which combined supervised fine-tuning and reinforcement learning. R1-Zero used rewards for correct answers and the required output format, allowing reasoning behaviour to emerge without an initial supervised fine-tuning stage [@deepseek-aiDeepSeekR1IncentivizingReasoning2025] ([original report](https://arxiv.org/html/2501.12948v1), sections 2.2 and 2.3).

LLM reasoning is distinct from [Agentic Reasoning](agentic-reasoning.md), which extends the process through [Tool Use](tool-use.md), [Planning](planning.md), [Reflection](reflection.md), [Memory](memory.md) and so forth. Reasoning LLMs can be part of an agentic system, but generating reasoning tokens alone does not make a system agentic.

Reasoning in human cognition is also distinct. The exact nature of how we reason is not fully understood. Humans can generalise from a few examples and intuitively recognise abstract relationships, but it would be too strong to say that LLMs have none of these capabilities. [Brown et al., 2020: Language Models are Few-Shot Learners](brown-et-al-2020-language-models-are-few-shot-learners.md) demonstrated that models can learn tasks from a few examples in context [@brownLanguageModelsFewShot2020]. Generating reasoning tokens does not establish that a model reasons in the same way as a human. That said, LLMs are able to do more and more of the things that were once considered uniquely human.

Finally, there is [Logical Reasoning](logical-reasoning.md), which allows us to arrive at conclusions from premises. More specifically, valid [Deduction](deduction.md) guarantees a true conclusion when its premises are true. Induction and abduction do not provide that guarantee. An LLM generating a chain of thought does not, by itself, guarantee a logically valid conclusion.

[Reasoning](../../../permanent/reasoning.md), more generally, is about getting to an answer through some intermediate process.
