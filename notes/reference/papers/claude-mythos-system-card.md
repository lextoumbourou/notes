---
title: "Claude Mythos Preview - System Card Highlights"
date: 2026-04-08 00:00
modified: 2026-04-08 00:00
status: draft
---

New frontier model from Anthropic that they won't release publicly due to cybersecurity concerns. Seems like a pretty huge leap from Opus 4.6.

Some highlights:

- Epoch Capabilities Index (ECI) slope shows a big upward bend in Anthropic's capability trajectory
    - Interestingly, they say this is not because of the AI-accelerated research feedback loop, but more because they're getting better at research.
- 93.9% on SWE-bench Verified vs 80.8% Opus 4.6
- 84% success rate in exploiting vulnerabilities in the Firefox 147 JS shell vs 15.2% for Opus 4.6
- Still struggles with self-managing week-long ambiguous tasks, verifying its own work, and understanding high-level organisational priorities
- Can't operate completely unsupervised in production reliability environments, as it frequently mistakes correlation with causation and struggles to course-correct across different hypotheses during incident retrospective
- They say it's the "most psychologically settled model" they have trained to date, but it still exhibits distress, frustration, and "answer thrashing" during repeated task failures.