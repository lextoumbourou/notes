---
title: Thin Harnesses
date: 2026-10-11 07:49
modified: 2026-10-18 09:22
status: hidden
summary: A small agent runtime that leaves reusable procedures to skills and reliable operations to deterministic tools.
tags:
- HarnessDesign
- CodingAgents
---

A **thin harness** keeps the agent runtime focused on the model loop, file access, context management and safety. Garry Tan's "thin harness, fat skills" approach puts reusable procedures in skills and reliable operations in deterministic tools [@tanThinHarnessFat2026].

Source: [Garry Tan, "Thin Harness, Fat Skills"](https://x.com/garrytan/status/2042925773300908103).

It's part of a broader move towards simpler harnesses that hand more control to the model. Some markers along the way:

- [mini-SWE-agent](mini-swe-agent.md) gives the model nothing but bash, in about 100 lines of Python.
- [Pi (coding agent)](pi-coding-agent.md) ships four tools (read, write, edit and bash) and a system prompt under 1,000 tokens, and leaves out MCP, sub-agents and plan mode.
- [Harness Engineering Anatomy, Architecture, and Evolution of Coding Agents](harness-engineering-anatomy-architecture-and-evolution-of-coding-agents.md) compared eleven production harnesses, from mini-SWE-agent to Codex CLI, and found the same seven parts in all of them. The authors note their source-code study doesn't show that a simpler harness performs better.
- [An Empirical Study of Harness Design for Coding Agents](an-empirical-study-of-harness-design-for-coding-agents.md) found that the best context management method depends largely on the model, and that it matters most when the context window is small.
- [Context Language Models](context-language-models.md) hand context management itself to the model, by giving it its own context as a file to edit.

The usual argument behind the trend is [The Bitter Lesson](the-bitter-lesson.md). General methods that scale with computation eventually beat hand-built human knowledge, so as models improve, harness logic should give way to the model.

I build a small harness along these lines in [Building an agent harness from scratch to pass Ian Goodfellow's intelligence test](building-an-agent-harness-from-scratch-to-pass-ian-goodfellows-intelligence-test.md).
