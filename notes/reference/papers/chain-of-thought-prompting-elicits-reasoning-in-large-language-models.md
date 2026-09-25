---
title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
date: 2024-09-29 00:00
modified: 2026-09-25 17:10
status: draft
category: paper
summary: Giving an LLM a few worked examples that show intermediate reasoning steps substantially improves its performance on maths, commonsense and symbolic reasoning tasks.
tags:
- LargeLanguageModels
- ReasoningModels
paper_title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
paper_url: https://arxiv.org/abs/2201.11903
paper_authors: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou
paper_year: 2022
doi: 10.48550/arXiv.2201.11903
---

This paper from Google, first released in January 2022, introduced [Chain-of-Thought Prompting](../../permanent/chain-of-thought-prompting.md). Instead of giving the model few-shot examples of questions and answers, you give it examples that show the working: a series of intermediate reasoning steps before the answer. The model then does the same for the new question [@weiChainofThoughtPromptingElicits2022].

No fine-tuning, no new architecture, just better examples. With eight chain-of-thought examples, PaLM 540B achieved state-of-the-art accuracy on the GSM8K benchmark of maths word problems, beating a fine-tuned GPT-3 with a verifier.

![Standard prompting versus chain-of-thought prompting. With standard examples the model answers 27, which is wrong. With worked examples it reasons 23 - 20 = 3, then 3 + 6 = 9, and answers 9 correctly.](../../_media/chain-of-thought-prompting-elicits-reasoning-in-large-language-models-fig-1.png)

*Figure 1: Standard prompting versus chain-of-thought prompting, from the paper.*

The catch is that it only works on big models. The authors describe chain-of-thought reasoning as an emergent ability of model scale. For small models it didn't help, and sometimes made things worse, because they produced fluent but illogical reasoning. The gains only showed up at around 100B parameters.

![Solve rates on GSM8K, SVAMP and MAWPS for LaMDA, GPT and PaLM at different model sizes. Chain-of-thought prompting only pulls ahead of standard prompting at the largest sizes.](../../_media/chain-of-thought-prompting-elicits-reasoning-in-large-language-models-fig-4.png)

*Figure 4: Chain-of-thought prompting only helps at large model scale.*

The same idea works beyond maths, including commonsense questions, date understanding and symbolic tasks like concatenating the last letters of words.

![Example chain-of-thought exemplars for nine benchmarks, including maths word problems, CSQA, StrategyQA, date understanding, sports understanding, SayCan, last letter concatenation and coin flip.](../../_media/chain-of-thought-prompting-elicits-reasoning-in-large-language-models-fig-3.png)

*Figure 3: Examples of input, chain of thought and output for arithmetic, commonsense and symbolic reasoning benchmarks.*

## Abstract

> We explore how generating a chain of thought, which is a series of intermediate reasoning steps, significantly improves the ability of large language models to perform complex reasoning.
>
> In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain-of-thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting.
>
> Experiments on three large language models show that chain-of-thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks.
>
> The empirical gains can be striking. For instance, prompting a PaLM 540B with just eight chain-of-thought exemplars achieves state-of-the-art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier.

See [LLM Reasoning](../../permanent/llm-reasoning.md) for what came next.
