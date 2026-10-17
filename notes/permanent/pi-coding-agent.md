---
title: "Pi (coding agent)"
date: 2026-10-18 09:22
modified: 2026-10-18 09:22
status: hidden
summary: "Mario Zechner's minimal open-source coding agent, with four tools and a system prompt under 1,000 tokens."
tags:
- CodingAgents
- HarnessDesign
---

[Pi](https://pi.dev/) is Mario Zechner's minimal open-source coding agent, built on the rule "if I don't need it, it won't be built" [@zechnerWhatLearnedBuilding2025].

It ships four tools by default: read, write, edit and bash. The system prompt and tool definitions together come in under 1,000 tokens, plus the user's `AGENTS.md` files. It deliberately leaves out MCP, sub-agents, plan mode and built-in to-do lists. The model keeps plans and tasks in files like `PLAN.md` and `TODO.md`, and can start another copy of itself through bash. Zechner reports competitive Terminal-Bench 2.0 results with Claude Opus 4.5.

The harness in [Building an agent harness from scratch to pass Ian Goodfellow's intelligence test](building-an-agent-harness-from-scratch-to-pass-ian-goodfellows-intelligence-test.md) follows its four-tool design. See also [Thin Harnesses](thin-harnesses.md).
