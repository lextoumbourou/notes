---
title: Speaker Disentanglement
date: 2023-12-01 00:00
modified: 2026-09-26 08:55
status: draft
---

**Speaker Disentanglement** is separating the speaker's identity (their timbre and other voice characteristics) from the content of what's being said in a speech representation.

It's the key step in most [Voice Conversion](voice-conversion.md) approaches: once content and speaker are separated, you can combine the content with a different speaker's characteristics. It's hard to do without losing some of the content, which is the problem [ContentVec](../reference/papers/papers-contentvec.md) tries to fix by modifying [HuBERT](hubert.md).
