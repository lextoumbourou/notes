---
title: Graph-enhanced Large Language Models in Asynchronous Plan Reasoning
date: 2024-10-11 00:00
modified: 2026-09-26 08:50
status: draft
---

## Overview

Paper investigates the ability of [Large Language Models](large-language-models.md) to perform [Asynchronous Planning](asynchronous-planning.md), which involves complex tasks that have sequential and parallel actions.

They develop a benchmark called Asynchronous WikiHow (AsyncHow), which contains 1.6K real-world tasks requiring both sequential and parallel actions.

They find that LLMs perform poorly without specific guidance on solving these tasks.

To address this, they introduce a novel prompting technique called [Plan Like a Graph](Plan%20Like%20a%20Graph.md) (PLaG), which incorporates graph representations into the prompts.

PLaG significantly improves LLM performance, but the study also reveals that LLMs struggle with complex planning tasks, highlighting limitations in their ability to simulate digital devices. The paper concludes by discussing the implications of these findings and potential future directions for research.