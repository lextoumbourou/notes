---
title: Research, Plan, Implement Workflow
date: 2026-03-22 00:00
modified: 2026-04-13 00:00
tags:
  - SoftwareEngineering
  - AgenticReasoning
status: draft
---

For people who have been using Claude Code or a similar agentic coding tool for a while, there's a very obvious pattern that we find ourselves doing: start a new session by asking the agent to research part of the code, then we tell it what we want and iterate on a plan, then we implement based on the plan.

This is absolutely not an original idea. My colleagues have been using this approach, and there's some great writing on it out there, like this blog post from Boris Tane [How I Use Claude Code](https://boristane.com/blog/how-i-use-claude-code)

The idea is that each development loop with Claude starts with **Research**, **Planning** then **Implementation**. It's very similar to [Spec-First LLM Development](spec-first-llm-development.md); in fact, it's just another term for it. However, it describes a concrete approach to using the tools, rather than an abstract idea.
##  Research

In the research phase, we ask the model to read the relevant code. In my work codebase, that's usually an entire service directory or a path within it. In Boris Tane's article, we write a persistence `research.md` file.  Although I do find that useful for some problems, I'm not sure the persistence is always required.

The key detail is that you want the agent to write a summary of its research, and check that it aligns with your understanding.

This research will be referred to during the next key stage.

## Planning

Next, we want to tell the LLM what we want to do and work together to develop a plan.

Claude Code has a Planning Mode that I sometimes use. t.

Usually, the outcome is a plan that includes a to-do list for the LLM.

### Planning Techniques

#### Interviews

Some useful techniques here are to get the LLM to interview you and to make sure you agree on exactly what the solution needs to look like.

#### Collaborate on Plan File

Boris likes to write the `plan.md`, which I have experimented with in the past. He adds inline notes directly to the plan and gets the model to read them. Repeating until the plan is correct.

## Implementation

Finally, we execute against the approved plan and mark tasks complete in the document. Getting the LLM to write tests first can be a great starting point. I like to work with the LLM here. Typically, it will not get everything right. There's no shame in editing code yourself, correcting comments and so forth.
 
---

Even though the pattern is obvious, it doesn't hurt to give it a name.
