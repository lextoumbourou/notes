---
title: Claude Fable 5.1
date: 2026-09-02 06:26
modified: 2026-09-23 06:08
status: draft
category: model
summary: "Anthropic's new frontier model targets long-running coding, knowledge work and scientific research."
tags:
  - ModelRelease
  - Claude
---

**Claude Fable 5.1** is Anthropic's new flagship model for coding, knowledge work and long-running agentic tasks. It is generally available today through Claude, the API, Amazon Web Services, Google Cloud and Microsoft Azure.

Anthropic also announced **Claude Mythos 5.1**, the same underlying model with more permissive cybersecurity and life-sciences safeguards. Mythos is limited to vetted organisations and professionals through Anthropic's trusted-access programs.

## What is new

Anthropic reports substantial gains over Fable 5 on agentic terminal coding, coding, scientific research and multidisciplinary reasoning. The more interesting claim is not simply a higher benchmark number: Fable 5.1 is meant to sustain long, unattended work while remaining readable and checking its own work.

That matters more than a flashy one-shot demo. The useful frontier model is the one that can investigate an unfamiliar codebase, form a plan, run tests, revise it when evidence disagrees and leave a human with an audit trail.

Anthropic's examples include diagnosing a rare production crash by disassembling an external vendor library, and a 38-hour machine-learning run that identified a label artefact, corrected it and ran several follow-up experiments. These are vendor-reported examples, not independent benchmarks, but they describe the direction that actually matters for coding agents.

## Pricing makes agentic work cheaper

Fable 5.1 retains Fable 5's standard API price of $10 per million input tokens and $50 per million output tokens. The material change is cache reads: they now cost $0.25 per million tokens, a 75% reduction.

That is a bigger deal than it sounds. Long-running agents repeatedly carry forward large contexts, tool traces and intermediate plans, so cache reads can make up much of their bill. Anthropic estimates about 25% lower cost for typical Fable workloads and as much as 45% lower cost for highly agentic work.

| Model | Input | Output | Cache read |
| --- | ---: | ---: | ---: |
| Claude Fable 5 | $10.00 | $50.00 | $1.00 |
| Claude Fable 5.1 | $10.00 | $50.00 | $0.25 |

*Prices are per million tokens. Anthropic's quoted savings depend on workload mix and how much cached context is reused.*

## A more useful safety boundary

Fable 5.1 can now identify software vulnerabilities, which Anthropic says reduces cybersecurity-safeguard interventions by about 60% per Claude Code session. It still redirects penetration testing, exploit generation and binary-based vulnerability scanning to more restricted models.

For sensitive work, Mythos 5.1 offers carefully broader access rather than an unrestricted release. Anthropic says its Cyber Verification Program will expand to Mythos-class models, while its Life Sciences Verification Program is being rolled out with US-government partners.

This is the right shape of trade-off. Defensive security work should not be crippled by a blunt classifier, but the boundary around genuinely dual-use work still needs to exist.

## Scientific capabilities, with appropriate caution

Anthropic also presents Fable and Mythos 5.1 as early evidence of AI systems contributing to scientific research. Its reported demonstrations include protein-binder design, a higher-resolution Venus elevation map, and GPU-kernel optimisations for open-source genomics and protein models.

Those claims are promising, especially the focus on measurable, externally validated results. They should still be read as launch claims, not a substitute for replication. The accompanying system card is the more valuable document: it describes the evaluation methods, residual risks and the safeguards that accompany the release.

## Sources

- [Introducing Claude Fable 5.1 and Claude Mythos 5.1](https://www.anthropic.com/claude-fable-and-mythos-5-1)
- [Claude Fable 5.1 and Claude Mythos 5.1 system card](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20%26%20Claude%20Mythos%205.1%20System%20Card.pdf)
