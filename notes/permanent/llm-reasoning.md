---
title: LLM Reasoning
date: 2025-07-28 00:00
modified: 2026-09-25 20:30
summary: How LLMs use intermediate reasoning tokens, from chain-of-thought prompting to trained reasoning models, and how this relates to other forms of reasoning.
cover: /_media/llm-reasoning-chess-knights-hassan-pasha.jpg
cover_credits: Photo by <a href="https://unsplash.com/@hpzworkz">Hassan Pasha</a> on <a href="https://unsplash.com/photos/black-and-white-chess-knights-7SjEuEF06Zw">Unsplash</a>
tags:
  - LargeLanguageModels
  - ReasoningModels
---

Reasoning, in the context of [LLMs](large-language-models.md), refers to the process of generating intermediate outputs before giving an answer. Typically, these outputs are tokens, which we call *reasoning* or *thinking* tokens. Reasoning in token-space is sometimes called [Chain-of-Thought Reasoning](chain-of-thought-reasoning.md) [@raschkaBuildReasoningModel2026].

However, LLMs don't only reason in token-space. [Latent Reasoning](latent-reasoning.md) is where the LLM performs intermediate reasoning in hidden representations rather than generating a text token for each step [@haoTrainingLargeLanguage2026].

The popularity of LLM reasoning seemed to explode after the release of the paper [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](../reference/papers/chain-of-thought-prompting-elicits-reasoning-in-large-language-models.md), in January 2022. The researchers showed that giving the model examples with intermediate reasoning steps substantially improved its ability on a range of arithmetic, commonsense and symbolic reasoning tasks [@weiChainofThoughtPromptingElicits2022], and it became a popular prompt engineering technique.

However, there had been papers exploring intermediate outputs in language models before that. In 2017, a DeepMind paper trained a model to generate step-by-step rationales before answering algebra word problems [@lingProgramInductionRationale2017]. In October 2021, OpenAI released the GSM8K dataset of maths word problems, fine-tuned GPT-3 on step-by-step solutions, and trained verifiers to pick the best of many candidate solutions [@cobbeTrainingVerifiersSolve2021]. A month later, the Scratchpads paper showed that language models could handle multi-step tasks like long addition and executing code if they were trained, or even just prompted, to write their intermediate working to a "scratchpad" before the final answer [@nyeShowYourWork2021].

What made chain-of-thought prompting different was that it needed no fine-tuning and worked across many kinds of tasks - it was an [Emergent Property](emergent-abilities.md) of sufficiently large LLMs, and a handful of worked examples in the prompt were enough to get a large model to show its working.

Shortly after the chain-of-thought paper, in March 2022, the [Self-Consistency](../reference/papers/self-consistency-improves-chain-of-thought-reasoning-in-language-models.md) paper introduced a new decoding strategy for chain-of-thought prompting. Instead of greedily decoding a single chain of thought, the model samples several diverse reasoning paths and the most common final answer is selected, effectively a majority vote. The intuition is that a hard problem can often be solved in several different ways that all lead to the same correct answer. It improved chain-of-thought results on arithmetic and commonsense benchmarks, including a 17.9 percentage point gain on GSM8K [@wangSelfConsistencyImprovesChain2022].

Later that year, [Large Language Models are Zero-Shot Reasoners (May 2022)](../reference/papers/large-language-models-are-zero-shot-reasoners-may-2022.md) showed that simply prompting the model with "Let's think step by step" before giving an answer could substantially improve its performance on a range of maths and logic tasks, without providing worked examples [@kojimaLargeLanguageModels2022].

A few years later, in late 2024, OpenAI released their o1 family of reasoning models, which were among the first examples of models trained with large-scale reinforcement learning to reason using chain of thought [@openaiOpenAIO1System2024]. The first releases, o1-preview and o1-mini, arrived in September 2024. The o1 API release in December 2024 added a reasoning-effort control, allowing developers to influence how much reasoning the model performs [@openaiO1Model] [@openaiAPIChangelog]. o1 popularised the [Test-Time Scaling](test-time-scaling.md) paradigm, allowing the models to spend a configurable amount of computation when answering a question to improve the result.

In January 2025, DeepSeek released R1 and shared a training recipe, allowing the world to understand in more detail how a reasoning model can be trained. Their paper, [DeepSeek-R1 Reasoning via Reinforcement Learning](../reference/papers/deepseek-r1-reasoning-via-reinforcement-learning.md), distinguishes [DeepSeek-R1-Zero](deepseek-r1-zero.md), which applied reinforcement learning directly to a pretrained base model, from R1, which combined supervised fine-tuning and reinforcement learning. R1-Zero used rewards for correct answers and the required output format, allowing reasoning behaviour to emerge without an initial supervised fine-tuning stage [@deepseek-aiDeepSeekR1IncentivizingReasoning2025] ([original report](https://arxiv.org/html/2501.12948v1), sections 2.2 and 2.3).

LLM reasoning is distinct from [Agentic Reasoning](agentic-reasoning.md), which extends the process through [Tool Use](tool-use.md), [Planning](planning.md), [Reflection](reflection.md), [Memory](memory.md) and so forth. Reasoning LLMs can be part of an agentic system, but generating reasoning tokens alone does not make a system agentic.

Reasoning in human cognition is also distinct. The exact nature of how we reason is not fully understood. Humans can generalise from a few examples and intuitively recognise abstract relationships, but it would be too strong to say that LLMs have none of these capabilities. [Brown et al., 2020: Language Models are Few-Shot Learners](brown-et-al-2020-language-models-are-few-shot-learners.md) demonstrated that models can learn tasks from a few examples in context [@brownLanguageModelsAre2020a]. Generating reasoning tokens does not establish that a model reasons in the same way as a human. That said, LLMs can do more and more of what was once considered uniquely human.

Finally, there is [Logical Reasoning](logical-reasoning.md), which allows us to arrive at conclusions from premises. More specifically, valid [Deduction](deduction.md) guarantees a true conclusion when its premises are true. Induction and abduction do not provide that guarantee. An LLM generating a chain of thought does not, by itself, guarantee a logically valid conclusion.

[Reasoning](reasoning.md), more generally, is about getting to an answer through some intermediate process.
