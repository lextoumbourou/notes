---
title: "ReAct: Synergizing Reasoning and Acting in Language Models"
date: 2024-10-05 00:00
modified: 2026-09-23 08:12
status: draft
category: paper
summary: ReAct interleaves language-model reasoning traces with actions and observations so an agent can update its plan using information from the environment.
tags:
- AgenticReasoning
- LanguageModels
paper_title: "ReAct: Synergizing Reasoning and Acting in Language Models"
paper_url: https://arxiv.org/abs/2210.03629v3
paper_authors: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan and Yuan Cao
paper_year: 2023
doi: 10.48550/arXiv.2210.03629
code: https://github.com/ysymyth/ReAct
---

[ReAct](react-agent.md) combines two capabilities that are often studied separately: reasoning in language and acting in an external environment.

The model produces a reasoning trace, takes an action such as querying a knowledge base or manipulating an environment, receives an observation, and then reasons again. This gives the model a way to update its plan, handle unexpected results and gather information that was not in its original context.

The authors evaluate the approach on question answering, fact verification and interactive decision-making tasks. On HotpotQA and Fever, interacting with a Wikipedia API helps ground the reasoning and reduce hallucination. On ALFWorld and WebShop, ReAct improves success over the comparison methods while using only one or two in-context examples.

The important idea is the loop itself: **reason, act, observe, repeat**. The [ReAct](react-agent.md) describes this loop in the context of an agent harness.

## Sources

- [Paper on arXiv](https://arxiv.org/abs/2210.03629v3)
- [Project page and code](https://react-lm.github.io/)
