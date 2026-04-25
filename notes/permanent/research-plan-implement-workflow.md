---
title: Research, Plan, Implement Workflow
date: 2026-03-22 00:00
modified: 2026-04-20 19:33
tags:
  - SoftwareEngineering
  - AgenticReasoning
cover: /_media/research-plan-implement-workflow-cover.jpg
summary: "An approach to agentic software development that I use"
---

For people who have been using Claude Code or a similar agentic coding tool for a while, a very obvious pattern emerges: start each session by asking the agent to research the code, work on a plan (or spec), then implement it.

This is absolutely not an original idea. My colleagues have been using this approach, and there's some great writing on it, like [this blog post by Boris Tane](https://boristane.com/blog/how-i-use-claude-code) (and I'm sure there are many others).

The idea is that each development loop starts with **Research**, then **Planning**, then **Implementation**. It's very similar to [Spec-First LLM Development](spec-first-llm-development.md) - in fact, it's a concrete instantiation of it. Where spec-first is the abstract principle, this is the hands-on workflow.

## Research

In the research phase, you ask the model to read the relevant code. For my work codebase, that's usually an entire service directory or a path within it. For a smaller project, it might be the entire repository. The prompt might be like:

> "Read and understand the billing service in the `billing_service` directory. Pay close attention to how we handle cancelling monthly subscriptions."

In Tane's workflow, findings always get written to a persistent `research.md` file. I find that useful for complex problems, though for smaller tasks, a verbal summary in the chat can be enough. The important thing, either way, is that you read the output and verify your understanding before moving to planning.

## Planning

Once you're satisfied with the research, ask for a detailed implementation plan. For example:

> "Come up with a plan for adding the ability to pause a monthly subscription. The user will be able to specify how many months to pause for. Include the approach, file paths, code snippets showing the key changes, and any trade-offs."

Claude Code's **Planning Mode** does a great job of this. Alternatively, storing it in a `plan.md` file gives you full control; you can edit it in your editor, and it persists as a real artifact.

One trick that works well: if you've seen a good implementation of something similar in another codebase, share that code as a reference alongside the plan request. The StrongDM [Software Factory](software-factory.md) approach calls this [Gene Transfusion](https://factory.strongdm.ai/techniques/gene-transfusion). A bit dramatic for my tastes - but you get the idea.

A useful technique is to get the LLM to interview you: ask it to clarify requirements before writing the plan. This forces both of you to agree on exactly what the solution needs to look like before any code is written. This is a feature built into Claude Code's Planning Mode, and it works great.

Typically, you expect to iterate on the plan a few times to get it right. This is the time you really want to be paying attention with your neurons firing - getting the plan right can save you a lot of time downstream.

The plan will usually also contain an implementation checklist to be followed during the build.

## Implementation

When the plan is right, issue the implementation command:

> "Implement based on the plan. Write the tests first."

Sometimes, depending on the problem, I won't require Claude to write the tests first, but TDD is often a nice way to verify the plan.

At the end of the implementation cycle, I'll review the changes, and typically expect some further iterations. I'm not afraid to edit the code myself - especially comments. Sometimes, I'll edit via pseudo code and ask it to go back and implement it, if I think the implementation is too far off.

---

The pattern tends to come naturally after using agentic coding tools for a while, but it's helpful to give it a name.

---

Photo by [Vooglam Eyewear](https://unsplash.com/@vooglam_official?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/photos/desk-with-laptop-blueprints-and-tools-0dhIl78b__o?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)