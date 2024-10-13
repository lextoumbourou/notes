---
title: "Audiobox: Unified Audio Generation with Natural Language Prompts"
date: 2023-12-13 00:00
modified: 2023-12-13 00:00
status: draft
---

Audio is an essential part of our life, but creating it often requires expertise and is
time-consuming. Research communities have made great progress over the past
year advancing the performance of large scale audio generative models for a single
modality (speech, sound, or music) through adopting more powerful generative
models and scaling data.

However, these models lack controllability in several
aspects: speech generation models cannot synthesize novel styles based on text
description and are limited on domain coverage such as outdoor environments;
sound generation models only provide coarse-grained control based on descriptions
like “a person speaking” and would only generate mumbling human voices.

This paper presents Audiobox, a unified model based on flow-matching that is capable of generating various audio modalities.

We design description-based and examplebased prompting to enhance controllability and unify speech and sound generation paradigms.

We allow transcript, vocal, and other audio styles to be controlled independently when generating speech.

To improve model generalization with limited labels, we adapt a self-supervised infilling objective to pre-train on large quantities of unlabeled audio

Audiobox sets new benchmarks on speech and sound generation (0.745 similarity on Librispeech for zero-shot TTS; 0.77 FAD on AudioCaps for text-to-sound) and unlocks new methods for generating audio with novel vocal and acoustic styles.

We further integrate Bespoke Solvers, which speeds up generation by over 25 times compared to the default ODE solver for flow-matching, without loss of performance on several tasks. Our demo is available
at https://audiobox.metademolab.com/.

1 Introduction
