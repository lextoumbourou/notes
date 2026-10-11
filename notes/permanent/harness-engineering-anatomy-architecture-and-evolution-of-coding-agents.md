---
title: Harness Engineering Anatomy, Architecture, and Evolution of Coding Agents
date: 2026-10-11 00:00
modified: 2026-10-11 07:09
category: paper
summary: A source-code comparison of eleven agent harnesses and the seven parts they have in common.
tags:
- HarnessDesign
- CodingAgents
paper_title: "Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents -- A Source-Code Study of Eleven Systems"
paper_url: https://arxiv.org/abs/2609.00006v1
paper_authors: Paul Barbaste, Tristan Darrigol, Germain Vu and Tom Wiltberger
paper_year: 2026
doi: 10.48550/arXiv.2609.00006
aliases:
- "Barbaste et al., 2026"
---

The authors inspected the source code of eleven agent harnesses, from Mini-SWE-Agent to Codex CLI, to see how they are built [@barbasteHarnessEngineeringAnatomy2026]. They organise each harness into seven parts: the agent loop, LLM integration, tools, context management, safety and permissions, orchestration, and extensibility.

## Method

They compared pinned versions of the eleven codebases across those seven parts. They also compared earlier and later versions of eight harnesses to see how their designs changed.

## Evaluation

This is a source-code study. The authors did not run the harnesses on the same tasks or measure their performance against one another.

## Results

The harnesses differ greatly in size, but all need to make decisions about the same basic parts. In the inspected runtimes, the authors found no general-purpose agent framework and no vector search over code. Nine of the eleven support skills, and eight support MCP [@barbasteHarnessEngineeringAnatomy2026].

### The seven parts

Table 1 describes the parts and shows how small or elaborate each can be [@barbasteHarnessEngineeringAnatomy2026]:

| Part | What it does |
|---|---|
| Agent loop | Calls the model, runs requested actions, and decides when to stop or recover from an error. |
| LLM integration | Builds requests and handles provider protocols, model settings and responses. |
| Tools and actions | Defines what the agent can do, such as running commands or editing files. |
| Memory and context | Chooses what the model sees now and what survives across turns or sessions. |
| Safety and permissions | Limits, approves or isolates what the agent can run. |
| Orchestration | Coordinates sub-agents when a task needs them. It can be absent. |
| Extensibility | Lets users add instructions, skills, plugins or external tools. |

The interface and session storage sit across these parts. The paper does not count them as two more parts (Section 2.3) [@barbasteHarnessEngineeringAnatomy2026].

## Takeaways

The seven parts are a useful checklist for the [Building an agent harness from scratch to pass Ian Goodfellow's intelligence test](building-an-agent-harness-from-scratch-to-pass-ian-goodfellows-intelligence-test.md). The paper's recommendations in Section 16 suggest a few practical choices [@barbasteHarnessEngineeringAnatomy2026]:

- Start with a simple loop and a small tool set. Add tools or more elaborate turn handling in response to problems you observe.
- Keep the model interface small, but allow provider-specific settings. A common API does not make caching, reasoning controls and other model features identical.
- Discover project instruction files and compact old context before the window fills, while keeping recent turns.
- Make limits and permission decisions explicit. A small teaching example should say where sandboxing would be needed for untrusted commands.
- Keep orchestration optional until there is a task that benefits from parallel agents. Skills can add workflows without changing the core loop; MCP is useful for external integrations.

The authors' 90-line example is an illustrative starting point, not production code. Their source-code study does not establish that a simpler harness performs better, or that stronger models always need less harness code. It shows what existing systems built and gives design suggestions to test against your own tasks (Sections 15 and 16) [@barbasteHarnessEngineeringAnatomy2026].
