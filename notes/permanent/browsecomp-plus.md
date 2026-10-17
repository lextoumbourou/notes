---
title: "BrowseComp-Plus"
date: 2026-10-18 08:19
modified: 2026-10-18 09:09
status: hidden
summary: "A deep-research benchmark that replaces live web search with a fixed corpus of about 100K documents, so agents and retrievers can be compared fairly."
source: https://arxiv.org/abs/2508.06600v1
tags:
- Benchmark
- AgenticReasoning
---

A deep-research benchmark that replaces live web search with a fixed corpus of about 100K documents, so agents and retrievers can be compared fairly.

Source: [BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent](https://arxiv.org/abs/2508.06600v1).

It's built from OpenAI's BrowseComp, 1,266 hard-to-find fact questions answered by browsing the live web. Live search results change over time and are a black box, which makes runs hard to reproduce and compare. BrowseComp-Plus keeps 830 of the questions, each with human-verified evidence documents and mined hard negatives, giving a corpus of 100,195 documents.

Referenced in [Context Language Models](context-language-models.md).
