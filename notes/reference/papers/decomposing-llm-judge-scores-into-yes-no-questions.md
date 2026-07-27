---
title: "Decomposing LLM Judge Scores Into Yes/No Questions"
date: 2026-07-26 00:00
modified: 2026-07-28 00:00
summary: "An LLM-judge approach that brings interpretability and actionability to your scores."
cover: /_media/binary-questions-cover.png
category: paper
tags:
- LLMJudge
- LargeLanguageModels
aliases:
- "Ask, Don’t Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement"
paper_title: "Ask, Don’t Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement"
paper_url: https://arxiv.org/pdf/2606.27226
paper_authors: Sangwoo Cho, Kushal Chawla, Pengshan Cai, Zefang Liu, Chenyang Zhu, Shi-Xiong Zhang and Sambit Sahu
---

Most LLM judges output a single score for each criterion. However, a score can often be opaque and difficult to act on, especially at the higher end: what should you change to get a 5 instead of a 4.5?

LLMs can also be biased by writing style, word choice, and length, which can result in misaligned scores.

This paper introduces an LLM-judge evaluation framework called [BinEval](../../permanent/bineval.md), which addresses this by decomposing each high-level evaluation criterion, or **dimension**, into a series of binary (yes/no) questions. This gives you interpretable, actionable feedback about why the output passed or failed each dimension. Plus, because you aggregate the binary answers into a single number, you still get one clean score that's easy to communicate.

The BinEval system has three parts:

1. A meta-prompt that decomposes the task prompt into a set of questions per dimension.
2. An evaluator that answers each question independently, then aggregates the answers per dimension into a score.
3. An optimisation loop that uses question-level feedback to improve the evaluator and generator prompts.

![The BinEval pipeline: a task prompt is decomposed by a meta-prompt into binary questions grouped by dimension; an evaluator answers each yes/no independently; the answers are aggregated into per-dimension and overall scores; and an optimisation loop feeds question-level disagreements back to rewrite the prompts and the questions.](../../_media/ask-dont-judge/beval-pipeline.png)

*My own diagram of the BinEval pipeline.*

The framework focuses on text tasks like summarisation, dialogue, and instruction following, but in theory it could apply to any LLM-judge evaluation task. Beyond evaluation, they also demonstrate how question-level feedback can be used to optimise system prompts and even the questions themselves.

## Method

BinEval is motivated by prior work showing the effectiveness of decomposing complex tasks into simpler sub-problems, for example [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910) and [Decomposed Prompting](https://arxiv.org/abs/2210.02406).

### Question generation

The prompt that describes your task is called the **"task prompt"**.

The prompt that decomposes the criterion into questions is called the **"meta-prompt"**.

They generate the questions in two steps:

In the first step, they summarise the task prompt into an explicit set of requirements. Each requirement captures a specific expectation, which helps the model build a representation of the task.

In the second step, they decompose each requirement into binary questions.

The questions are grouped into broader evaluation dimensions (like coherence or fluency), which is what lets you score each dimension separately later. In other words, requirements are the specific expectations extracted from the task, while dimensions are the categories used to organise and report the resulting questions.

![Table 9 from the paper: the binary questions BinEval uses for the coherence dimension on SummEval](../../_media/ask-dont-judge/table9-coherence-binary-questions.png)

The meta-prompt is task-agnostic; that is, only the task prompt changes from one task to the next.

### Scoring

Here, $\textcolor{#0072B2}{x}$ is the source or input, $\textcolor{#D55E00}{y}$ is the output being evaluated, and $\textcolor{#009E73}{f_E}$ returns 1 for "yes" and 0 for "no". $\textcolor{#CC79A7}{q_i}$ is an individual question and $\textcolor{#CC79A7}{Q_d}$ is the set of questions for dimension $\textcolor{#CC79A7}{d}$, while $\textcolor{#E69F00}{N}$ is the total number of questions.

Since the questions are grouped by dimension, a dimension's score is just the fraction answered yes:

$$S_{\textcolor{#CC79A7}{d}}(\textcolor{#0072B2}{x}, \textcolor{#D55E00}{y}) = \frac{1}{|\textcolor{#CC79A7}{Q_d}|} \sum_{\textcolor{#CC79A7}{q_i} \in \textcolor{#CC79A7}{Q_d}} \textcolor{#009E73}{f_E}(\textcolor{#0072B2}{x}, \textcolor{#D55E00}{y}, \textcolor{#CC79A7}{q_i})$$

The overall score is calculated identically across all $N$ questions:

$$S(\textcolor{#0072B2}{x}, \textcolor{#D55E00}{y}) = \frac{1}{\textcolor{#E69F00}{N}} \sum_{i=1}^{\textcolor{#E69F00}{N}} \textcolor{#009E73}{f_E}(\textcolor{#0072B2}{x}, \textcolor{#D55E00}{y}, \textcolor{#CC79A7}{q_i})$$

And if you want to map the scores from $[0, 1]$ to some other interval $[a, b]$, you can do it via affine scaling:

$$S'(\textcolor{#0072B2}{x}, \textcolor{#D55E00}{y}) = S(\textcolor{#0072B2}{x}, \textcolor{#D55E00}{y}) \cdot (b - a) + a$$

## Experiments and Results

The experiments ask two questions. First, how closely do BinEval's scores agree with human ratings? Second, can its question-level feedback be used to improve prompts?

For evaluation quality, they check how well each method's scores correlate with human ratings (Spearman, Kendall, Pearson) across three benchmarks:

- [SummEval](../../permanent/summeval.md): summarisation, rated on fluency, coherence, consistency, and relevance (1-5).
- [Topical-Chat](../../permanent/topical-chat.md): dialogue responses, rated on qualities like naturalness, coherence, engagingness, and groundedness.
- [QAGS](../../permanent/qags.md): factual consistency (hallucination) in summaries.

They compare BinEval against a spread of traditional and model-based metrics:

- [ROUGE-1](../../permanent/rouge-1.md): a simple overlap metric.
- [BERTScore](../../permanent/bertscore.md): matches contextual BERT token embeddings.
- [MoverScore](../../permanent/moverscore.md): similar to BERTScore, but uses Earth Mover's Distance.
- [BARTScore](../../permanent/bartscore.md): scores text by how likely a language model is to generate it.

They also compare it against two LLM-judge approaches:

- [UniEval](../../permanent/unieval.md): an LLM-judge precursor to BinEval that asks a single yes/no per dimension.
- [G-Eval](../../permanent/g-eval.md): an LLM-judge approach that scores via chain-of-thought, but still gives one opaque number.

### Overall performance

Overall, BinEval achieves the best average correlation with human ratings on SummEval and Topical-Chat, with its strongest gains appearing on factual consistency. Its main weakness is relevance.

![Table 1 from the paper: summary-level Spearman / Kendall correlations on SummEval. BinEval (Claude) has the best average (0.563 / 0.491), leading on coherence, consistency, and fluency, while G-Eval (GPT-4) leads on relevance.](../../_media/ask-dont-judge/table1-summeval-correlations.png)

On SummEval, BinEval (Claude) has the best average correlation with human ratings ([Spearman Correlation](../../permanent/spearman-correlation.md) and Kendall), winning coherence, consistency, and fluency. And it transfers: on Topical-Chat it again gets the best average, even on subjective dialogue qualities like naturalness and engagingness.

### Where decomposition helps most

One clean win is **factual consistency**. On QAGS, splitting "is this faithful to the source?" into several targeted yes/no questions achieves the best average rank correlations, although it only narrowly beats G-Eval (GPT-4). Consistency is also where it gains the most on SummEval.

Breaking a broad judgement into concrete checks appears to be where decomposition pays off most. Each question is easier to answer in isolation, while averaging across several answers may reduce noise. The question set can also explicitly cover distinct failure modes that a holistic judgement might overlook.

### Better score behaviour

BinEval tracks the *shape* of human scores more faithfully and avoids the ceiling effects that squash prior judges into a narrow band, so it draws sharper distinctions between borderline and clearly bad outputs. Even on a weaker backbone (gpt-oss), it beats G-Eval and UniEval, where UniEval's single yes/no collapses on fluency (near-zero correlation), a sign that one question per dimension is too coarse.

![Score distributions on SummEval: BinEval more closely follows the distribution of human ratings, while other LLM judges cluster their scores toward the top of the scale.](../../_media/ask-dont-judge/figure1-summeval-distributions.png)

### Interpretability

Because every answer is kept, you can see *which* questions failed, not just a final number. In the paper's factual-consistency case study, you can inspect the source and summary, then see the specific checks behind the score.

![A source excerpt and the summary evaluated against it, from the paper](../../_media/ask-dont-judge/source-summary-example.png)

![BinEval's factual-consistency case study: individual binary questions expose which claims in the summary are and are not supported by the source.](../../_media/ask-dont-judge/figure4-consistency-case-study.png)

### Where decomposition struggles

The clear weak spot is **relevance**: G-Eval (GPT-4) still beats it there, and it's the one dimension that didn't improve under the iterative update either. A broad, holistic judgement like "is this relevant to the source?" doesn't cleanly break into independent yes/no checks, so decomposition loses its edge. The results suggest a broader pattern: concrete criteria appear to decompose more reliably than broad, holistic ones.

### Improving prompts with the feedback

So far, BinEval has been used to evaluate outputs. The second experiment asks whether its question-level failures can also be used to improve the prompts behind the evaluator or generator. They try two modes on SummEval:

- **Self-update**: one model (`gpt-oss-120b`) improves its own evaluator prompt from its failures against human ratings.
- **Cross-model update**: a stronger model (Claude Sonnet 4) acts as the reference, and disagreements are used to update the weaker target model's prompt.

Both improved three of the four dimensions (fluency +0.119 via self-update, consistency +0.136 via cross-model update), with relevance again the holdout. They also apply the loop to generation prompts on IFBench, an instruction-following benchmark with programmatically verifiable outputs. The catch is that iteration only helps when the failure is unclear instructions, not a capability ceiling. If the model simply can't do it, better questions won't save it.

## Summary

Decomposition is a promising approach for people looking for LLM-judge scores that are interpretable and actionable. The results suggest that it works best on concrete criteria like factual consistency and not as well on broad, holistic ones like relevance. Even where it isn't the single best score, it remains substantially more interpretable than a single holistic score.

The question-level failures can also be used to improve evaluator and generator prompts, provided the problem is unclear instructions rather than a capability ceiling. The trade-off is extra evaluation work, and the quality of the result still depends on the quality of the generated questions.

---

*Cover: a photo of a sculpture by Swiss artist Markus Raetz.*
