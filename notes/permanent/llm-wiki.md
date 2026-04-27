---
title: LLM Wiki
date: 2026-04-20 18:59
modified: 2026-04-20 18:59
tags:
- KnowledgeManagement
status: draft
---

**LLM Wiki** is a pattern for building personal knowledge bases using LLMs, proposed by [Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) (April 2026).


Instead of RAG (re-deriving knowledge every query), the LLM **incrementally builds and maintains a persistent wiki** - structured, interlinked markdown files between you and the raw sources. Knowledge is compiled once and kept current, not re-derived on every query.

> "The wiki is a persistent, compounding artifact. The cross-references are already there. The contradictions have already been flagged. The synthesis already reflects everything you've read."
