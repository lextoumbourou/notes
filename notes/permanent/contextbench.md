---
title: "ContextBench"
date: 2026-10-18 09:16
modified: 2026-10-18 09:16
status: hidden
summary: "A diagnostic benchmark of four synthetic tasks that tests context management on its own, separate from reasoning and knowledge."
tags:
- Benchmark
- ContextManagement
---

A diagnostic benchmark of four synthetic tasks that tests [Context Management](context-management.md) on its own, separate from reasoning and knowledge. It was introduced in the [Context Language Models](context-language-models.md) paper [@shaoContextLanguageModels2026].

The input arrives as a stream of operations, one user message each, so everything lands in the agent's context before it can act on it. Over an episode the stream adds up to as much as 24 times the 32K context limit, which the authors call context pressure. The tasks need no search or inference, so an agent that could keep everything would answer every query.

- **Needle Retention** asks the agent to keep specific lines word for word while deleting filler.
- **Sudoku Sketchpad** asks it to keep a 16×16 Sudoku board up to date as moves stream in, one per turn.
- **KV Store** sends batches of 100 key-value pairs, then asks for exact values later.
- **Log Triage** streams service logs, then asks lookup and count questions about them.

[![Four panels illustrate the ContextBench tasks: Needle Retention keeps highlighted lines from several pages, Sudoku Sketchpad edits cells in place on a board, KV Store answers a GET for one key out of many SETs, and Log Triage counts ERROR lines in a log. Below each panel, a line chart plots accuracy against context pressure for Base, RLM, Summary, Context folding, Self-Compact, ACM and CLM. CLM stays at 1.0 across every pressure level on all four tasks, while the other methods fall away as pressure rises.](../_media/context-language-models/contextbench-figure-2.png)](../_media/context-language-models/contextbench-figure-2.png)

*Figure 2 from [@shaoContextLanguageModels2026], CC BY 4.0. All methods use GPT-5.4 with a 32K context limit.*

With GPT-5.4, none of the existing methods (Codex-style summary, Context Folding, RLM, Self-Compact and ACM) were perfect, even on these simple tasks. Summaries lose or hallucinate needles and Sudoku cells, and coding tools can offload values to disk but can't evict them from the live context. CLM stays near 100% at every level of pressure.

Every method gets a task-specific skill explaining how to use its own tools on each task, so ContextBench tests whether a method can carry out a context management strategy, not whether the model can come up with one.
