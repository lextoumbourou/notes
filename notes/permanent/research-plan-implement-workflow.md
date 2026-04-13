---
title: Research, Plan, Implement Workflow
date: 2026-03-22 00:00
modified: 2026-04-13 00:00
tags:
  - SoftwareEngineering
  - AgenticReasoning
status: draft
---

For people that have been using Claude Code or Codex for a while, there's a very obvious pattern that we find ourselves doing: we ask Claude to research part of the code, then we tell it what we want and ask for a plan that we iterate on, then we implement based on the plan.

This is absolutely not an original idea. My colleagues have been using this approach, and there's a some great writing on it out there, like this blog post from Boris Tane [How I Use Claude Code](https://boristane.com/blog/how-i-use-claude-code)

The idea is that each development loop with Claude starts with **Research**, **Planning** then **Implementation**. It's very similar to [Spec-First LLM Development](spec-first-llm-development.md) , in fact, it is really just another term for it. However, it describes a concrete approach to using the tools, rather than an abstract idea.
##  Research

   - Ask the model to deeply read relevant code and write a persistent `research.md`.
    * study the notification system in great details, understand the intricacies of it and write a detailed research.md document with everything there is to know about how notifications work
    * read this folder in depth, understand how it works deeply, what it does and all its specificities. when that’s done, write a detailed report of your learnings and findings in research.md
   - Purpose: verify understanding before making architectural changes
## Planning

Generate a detailed `plan.md` with file-level changes, snippets, and trade-offs.

I want to build a new feature `<name and description>` that extends the system to perform `<business outcome>`. write a detailed plan.md document outlining how to implement this. include code snippets

## **Annotation cycle**

   - Human adds inline notes directly in the plan.
   - Model updates the plan to address notes.
   - Repeat until the plan is correct.
#### **Todo breakdown**
   - Add granular checklist/tasks to track progress during execution.

#### **Implementation**

   - Execute against the approved plan and mark tasks complete in the document.
   - Keep quality constraints explicit (types, checks, no unnecessary bloat).
   
### **Terse feedback loop**
   - During implementation, human gives short corrections and references existing patterns.


- Separates **thinking** from **typing**.
- Uses markdown plans as shared, durable state (survives long sessions/compaction better than chat alone).
- Reduces early wrong assumptions and rework.
- Keeps human judgment in control of scope and trade-offs.

## Connection to Spec-First LLM Development


This is strongly aligned with [[public/notes/permanent/spec-first-llm-development|Spec-First LLM Development]]: both use persistent specs/plan artifacts as the control surface for AI-assisted coding before implementation.

## Practical takeaway
Treat implementation as a mostly mechanical phase. Put creativity and decision-making into research + plan review iterations first.
