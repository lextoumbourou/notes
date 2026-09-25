---
title: "ToolGen: Unified Tool Retrieval and Calling via Generation"
date: 2024-10-11 00:00
modified: 2026-09-25 17:19
status: draft
category: paper
tags:
- AgenticReasoning
- LargeLanguageModels
paper_title: "ToolGen: Unified Tool Retrieval and Calling via Generation"
paper_url: https://arxiv.org/abs/2410.03439
paper_authors: Renxi Wang, Xudong Han, Lei Ji, Shu Wang, Timothy Baldwin, Haonan Li
paper_year: 2024
---

A [Tool Use](../../permanent/tool-use.md) paper, first released in October 2024, that represents each tool as a unique token in the LLM's vocabulary.

The usual approach is to put tool descriptions in the prompt, which is limited by context length and needs a separate retrieval step to pick which tools to include. ToolGen instead trains the tool knowledge into the model's parameters, so picking a tool and calling it are just part of next-token prediction. No separate retriever needed.

They test it with over 47,000 tools and report better results for both tool retrieval and completing tasks autonomously.

![Title, authors and abstract of ToolGen: Unified Tool Retrieval and Calling via Generation.](../../_media/toolgen-abstract.png)
