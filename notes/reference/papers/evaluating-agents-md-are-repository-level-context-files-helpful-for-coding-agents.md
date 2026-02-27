---
title: "Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?"
date: 2026-02-26 00:00
modified: 2026-02-26 00:00
summary: "Self-generated agent context files don't help."
cover: /_media/evaluating-agents-md-page-1.png
tags:
- AgenticReasoning
- SoftwareEngineering
- AIAgents
category: reference/papers
---

Self-Generated [Agent Context Files](../../../../permanent/agent-context-files.md) don't help either.

[This paper](https://arxiv.org/abs/2602.11988 ), released around the same time as [SkillsBench](../../permanent/skillsbench.md), evaluates whether repository context files (`AGENTS.md`, `CLAUDE.md`) actually improve coding-agent performance.

The authors test issue-resolution tasks (bug fixes and feature work derived from real PRs) under three settings:

- no context file
- LLM-generated context file
- developer-written context file

Similarly to SkillsBench's results, they conclude that LLM-generated context files generally reduce task success rates (about 3% on average) while increasing costs by 20%+. [^1]

Developer-written context files perform better than LLM-generated ones and can provide a small lift (~4% on average), but they still increase token usage. [^1]

Seems that agents do follow context-file instructions, but it typically means they run more tests, traverse more files, and spend more reasoning tokens. The problem is that this is often just extra exploration overhead rather than a meaningful improvement. Also, if the repository already has good developer documentation, duplicating the information into agent content files adds little value - agents can already read the existing docs.

So, putting the paper findings together, it seems the rules of thumb with context files and agents are:

- don't mindlessly auto-generate AGENTS files
- focus effort on standard high-quality developer documentation first
- keep agent context files minimal and high-signal
- use a small number of targeted skills/context instructions where they clearly help

Related articles:

- [SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks](./skillsbench-benchmarking-how-well-agent-skills-work-across-diverse-tasks.md)
- [OpenClaw: the missing piece for Obsidian's second brain](../../permanent/openclaw-the-missing-piece-for-obsidians-second-brain.md)
- [Spec-First LLM Development](../../permanent/spec-first-llm-development.md)

[^1]: Gloaguen, T., Li, J., Schmid, L., Bichsel, B., and Vechev, M. (2026). *Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?* arXiv. [https://arxiv.org/abs/2602.11988](https://arxiv.org/abs/2602.11988 )