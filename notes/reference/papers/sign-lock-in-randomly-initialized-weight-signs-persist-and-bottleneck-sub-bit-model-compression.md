---
title: "Sign Lock-In: Randomly Initialised Weight Signs Persist and Bottleneck Sub-Bit Model Compression"
date: 2026-02-27 00:00
modified: 2026-02-27 00:00
summary: "a heuristic for assigning signs to weights to make it easy to compress."
cover: /_media/sign-lock-in-page-1.png
status: draft
tags:
- MachineLearning
- ModelCompression
- Quantisation
- AIAgents
category: reference/papers
---

This paper focuses on improving the compression of the sign bit (+/-) in weight matrices.

Trained sign matrices look almost like random noise (hard to compress to low rank), yet the authors show they barely change during training. Most signs stay aligned with initialisation, and flips tend to happen only through rare crossings near zero. The authors call this behaviour "sign lock-in".

Across MLPs, CNNs, and Transformers, they show that sign patterns are much less compressible than magnitudes. However, the sign spectra look close to noise, and sign trajectories are consistently - they very rarely change.

They then formalise this with a stopping-time-style theory: under bounded updates and rare re-entry into a small neighbourhood around zero, effective sign flips follow a geometric tail pattern.

From there, they propose two practical interventions to preserve sign structure for sub-bit compression:

- **gap-based initialisation** (start farther from the sign boundary)
- **outward-drift regularisation** early in training (discourage repeated near-zero crossings)

Empirically, these reduce effective sign flips to around **10^-3** with only about a **~1 point perplexity** increase.[^1]

The useful mental model: sub-bit compression is no longer mostly about magnitude-coding tricks; it becomes a **sign management problem**. If signs remain random-looking and unstructured, you hit the paper's "one-bit wall." If you can preserve or impose a reusable sign structure, you have a better chance of pushing below it.

Related articles:

- [Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?](./evaluating-agents-md-are-repository-level-context-files-helpful-for-coding-agents.md)
- [SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks](./skillsbench-benchmarking-how-well-agent-skills-work-across-diverse-tasks.md)

[^1]: Sakai, A., & Ichikawa, Y. (2026). *Sign Lock-In: Randomly Initialised Weight Signs Persist and Bottleneck Sub-Bit Model Compression* (v1). arXiv. https://arxiv.org/abs/2602.17063