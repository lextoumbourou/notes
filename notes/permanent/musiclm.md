---
title: MusicLM
date: 2026-09-26 08:55
modified: 2026-09-26 08:55
status: draft
---

**MusicLM** is a text-to-music model from Google that generates music from text descriptions like "a calming violin melody backed by a distorted guitar riff".

It builds on [AudioLM](audiolm.md), modelling music as a hierarchy of discrete audio tokens: semantic tokens for long-term structure, and acoustic tokens from the [SoundStream](soundstream.md) codec for the fine audio details.

From paper [MusicLM: Generating Music From Text](../reference/papers/paper-musiclm-generating-music-from-text.md).
