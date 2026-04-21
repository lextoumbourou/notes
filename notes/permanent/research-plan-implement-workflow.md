---
title: Research, Plan, Implement Workflow
date: 2026-03-22 00:00
modified: 2026-04-20 19:33
tags:
  - SoftwareEngineering
  - AgenticReasoning
status: draft
---

For people who have been using Claude Code or a similar agentic coding tool for a while, a very obvious pattern emerges: start each session by asking the agent to research the code, work on a plan (or spec), then implement it.

This is absolutely not an original idea. My colleagues have been using this approach, and there's some great writing on it, like [this blog post by Boris Tane](https://boristane.com/blog/how-i-use-claude-code) (and I'm sure there are many others).

The idea is that each development loop starts with **Research**, then **Planning**, then **Implementation**. It's very similar to [Spec-First LLM Development](spec-first-llm-development.md) - in fact, it's really just a concrete instantiation of it. Where spec-first is the abstract principle, this is the hands-on workflow.

## Research

In the research phase, you ask the model to read the relevant code. For my work codebase, that's usually an entire service directory or a path within it. For a smaller project, it might be the entire repository.

In Tane's workflow, findings always get written to a persistent `research.md` file. I find that useful for complex problems, though for smaller tasks, a verbal summary in the chat can be enough. The important thing, either way, is that you read the output and verify your understanding before moving to planning.

As Tane notes, the most expensive failure mode with AI-assisted coding isn't wrong syntax but implementations that work in isolation but break the surrounding system. A function that ignores an existing caching layer. A migration that doesn't account for the ORM's conventions. An API endpoint that duplicates logic that already exists. The research phase prevents all of this.

## Planning

Once you're satisfied with the research, ask for a detailed implementation plan. Claude Code's **Planning Mode** does a great job of this. Alternatively, storing it in a  `plan.md` file gives you full control; you can edit it in your editor, and it persists as a real artifact.

> I want to build a new feature `<name and description>` that extends the system to perform `<business outcome>`. Write a detailed plan.md document outlining how to implement this. include code snippets

The plan should include the approach, code snippets showing actual changes, file paths that will be modified, and any trade-offs or considerations.

One trick that works well: if you've seen a good implementation of something similar in another codebase, share that code as a reference alongside the plan request. Claude works dramatically better when it has a concrete reference to work from rather than designing from scratch.

### Planning Techniques

There are some other common Planning Techniques I use and have seen others use.

#### Interviews

A useful technique is to get the LLM to interview you: ask it to clarify requirements before writing the plan. This forces both of you to agree on exactly what the solution needs to look like before any code is written.

#### Annotation Cycle

This is the most distinctive part of Tane's workflow, and where the human adds the most value.

After Claude writes the plan, you open it in your editor and add inline notes directly into the document. These notes correct assumptions, reject approaches, add constraints, or provide domain knowledge that Claude doesn't have:

- "use drizzle:generate for migrations, not raw SQL" - domain knowledge
- "no - this should be a PATCH, not a PUT" - correcting an assumption
- "Remove this section entirely, we don't need caching here" - rejecting an approach
- "The queue consumer already handles retries, so this retry logic is redundant" - explaining why

Then send Claude back to the document:

I added a few notes to the document, addressed all of them, and updated it accordingly. don't implement yet

This cycle repeats 1–6 times. The explicit **"don't implement yet"** guard is essential — without it, Claude will jump to code the moment it thinks the plan is good enough.

The markdown file acts as shared mutable state. You can think at your own pace, annotate precisely where something is wrong, and re-engage without losing context. This is fundamentally different from steering implementation through chat messages — a plan is a structured specification you can review holistically, while a chat conversation is something you'd have to scroll through to reconstruct the decisions.

Three rounds of annotation can transform a generic implementation plan into one that fits perfectly into the existing system.

#### Todo Breakdown

Before implementation starts, request a granular task breakdown:

> add a detailed to-do list to the plan, with all the phases and individual tasks necessary to complete the plan - don't implement yet

This creates a checklist that serves as a progress tracker. Claude marks items as completed as they go, so you can glance at the plan at any point and see where things stand.

## Implementation

When the plan is right, issue the implementation command. I've settled on something like:

> implement it all. When you're done with a task or phase, mark it as completed in the plan document. Do not stop until all tasks and phases are completed. Do not add unnecessary comments or JS docs; do not use any unknown or unsupported types. continuously run typecheck to make sure you're not introducing new issues.

By this point, every decision has been made and validated. Implementation should be boring, that's deliberate. The creative work happened in the annotation cycles.

Without the planning phase, what typically happens is that Claude makes a reasonable but wrong assumption early on, builds on it for 15 minutes, and then you have to unwind a chain of changes. The planning phase eliminates this entirely.

Getting the LLM to write tests first can be a great starting point. And there's no shame in editing code yourself, correcting comments and so forth — you're still the developer.

### Terse Feedback Loop

During implementation, your role shifts from architect to supervisor. Corrections become short:

- "wider"
- "still cropped"
- "There's a 2px gap"
- "I reverted everything. Now all I want is..."

Claude has the full context of the plan, so terse corrections are enough. For visual issues, attach screenshots. For consistency, reference existing code: "this table should look exactly like the users table."

Treat implementation as a mostly mechanical phase. Put creativity and decision-making into research + plan review iterations first. By the time you say "implement it all," the hard work should already be done.

Even though the pattern is obvious, it doesn't hurt to give it a name.