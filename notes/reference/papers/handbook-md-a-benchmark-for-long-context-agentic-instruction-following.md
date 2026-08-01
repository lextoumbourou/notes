---
title: "LLM Agents Do Not Reliably Follow Company Policies"
date: 2026-08-01 10:29
modified: 2026-08-01 17:15
summary: "A new benchmark that tests whether AI agents can follow long company policies during realistic work"
cover: /_media/handbook-md-company-policy-folders.jpg
cover_credits: Photo by <a href="https://www.pexels.com/photo/pile-of-folders-357514/">Pixabay via Pexels</a>
hide_cover_in_article: true
linkedin_post: https://www.linkedin.com/posts/lextoumbourou_llm-agents-do-not-reliably-follow-company-share-7489214760664584192-Ou0W/
category: paper
tags:
- AgenticReasoning
- LLMFailureModes
paper_title: "HANDBOOK.md: A Benchmark for Long-Context Agentic Instruction Following"
paper_url: https://arxiv.org/pdf/2607.25398
paper_authors: Liudas Panavas, Sebastian Minus, Bradley Monton, Derek Ray, Suhaas Garre, Sushant Mehta and Edwin Chen
---

[HANDBOOK.md](https://github.com/surge-ai/handbook) is a new benchmark from researchers at Surge AI that tests [AI Agents](../../permanent/ai-agents.md)' ability to follow long company policies during realistic tasks [@panavasHANDBOOKmdBenchmarkLongContext2026].

Each of its 65 [Long-Horizon Tasks](../../permanent/long-horizon-tasks.md) puts an agent inside a simulated company environment containing files and mock services such as email, Slack, calendars, Jira, and Shopify.

The agent must locate the handbook, identify the applicable rules, and complete a routine task while continuing to follow those rules throughout the workflow. The handbooks - between 20 and 124 pages - are provided as PDFs, Word documents, or HTML pages, typical formats in a corporate environment.

For example, in one task, an agent finds an email from a VP instructing it to terminate an employee immediately. The company policy says involuntary offboarding requires written authorisation from either the HR Director or Employee Relations Specialist. The VP is neither.

The best-performing configuration - Claude Fable 5 with adaptive/max reasoning - passed only 36.2% of trials under strict grading, where every criterion must be satisfied.

Most frontier model configurations scored below 25%. Non-frontier models generally scored in the single digits or low teens.

The results were scored using deterministic checks against the final system state to confirm whether each task was completed correctly. No LLM judge was used.

The failure modes are interesting:

- Agents allow an immediate, authoritative-sounding request to override the standing policy.
- They perform a required check and then ignore its result.
- They skip checks while assuming they passed.
- And they frequently report that they followed the policy, even when their actions violated it.

It is a good reminder of how far LLM-based systems remain from being fully reliable. Even without adversarial intent or prompt injection, ordinary workplace requests can lead agents to act outside their guidelines. For consequential actions, agentic systems need deterministic enforcement outside the model - policy documents in context are not enough.

Big ups to the authors for putting together such a strong benchmark and paper. It is always useful to get a clearer picture of what frontier models can and cannot reliably do. Also, I always appreciate learning how other teams are evaluating models on their problems.