---
title: Experience Replay
date: 2025-05-05 00:00
modified: 2026-09-26 08:55
status: draft
---

**Experience Replay** is a technique in [Reinforcement Learning (RL)](reinforcement-learning.md) where an agent stores its past transitions (state, action, reward, next state) in a buffer, then trains on random mini-batches sampled from that buffer.

Sampling randomly breaks the correlation between consecutive experiences, and lets the agent learn from each experience many times, which makes training more stable and data-efficient.

It was a key ingredient of [Deep-Q Learning](deep-q-learning.md), described in [Playing Atari with Deep Reinforcement Learning](../reference/papers/playing-atari-with-deep-reinforcement-learning.md).
