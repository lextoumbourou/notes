---
title: Voice Conversion
date: 2023-12-01 00:00
modified: 2026-09-26 08:50
status: draft
---

**Voice Conversion** is the task of transforming a recording of one speaker so that it sounds like it was spoken by a different target speaker, while keeping the linguistic content (what was said) the same.

See [Voice Conversion](https://paperswithcode.com/task/voice-conversion).

Most approaches try to separate the content of the speech from the speaker's identity (see [Speaker Disentanglement](speaker-disentanglement.md)), then combine the content with the target speaker's characteristics and turn it back into audio with a vocoder like [HiFi-GAN](hifigan.md). It's closely related to [Speech Synthesis](speech-synthesis.md), except the input is speech rather than text.

I played around with voice conversion in [Making Song Covers With My AI Voice](making-ai-covers-with-my-own-voice.md).
