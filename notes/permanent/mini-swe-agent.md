---
title: "mini-SWE-agent"
date: 2026-10-18 09:22
modified: 2026-10-18 09:22
status: hidden
summary: "A minimal coding agent from the SWE-agent team that gives the model nothing but bash."
tags:
- CodingAgents
- HarnessDesign
---

A minimal coding agent from the [SWE-agent](swe-agent.md) team that gives the model nothing but bash [@sweagentMiniSWEAgent].

The agent class is about 100 lines of Python. It doesn't use the model's tool-calling interface at all. The model writes a bash command, and each one runs independently with `subprocess.run`, so swapping in `docker exec` is enough to sandbox it. The history is strictly linear, so the messages the model sees are the trajectory itself. Its README claims over 74% on SWE-bench Verified.

It's one of the eleven harnesses compared in [Harness Engineering Anatomy, Architecture, and Evolution of Coding Agents](harness-engineering-anatomy-architecture-and-evolution-of-coding-agents.md), and the shared backbone for every method in the [Context Language Models](context-language-models.md) experiments. See also [Thin Harnesses](thin-harnesses.md).
