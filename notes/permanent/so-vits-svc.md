---
title: So-VITS-SVC
date: 2023-12-13 00:00
modified: 2023-12-13 00:00
status: draft
---

So-VITS-SVC is an open-source project that has spawned many, many forks. It is a [Singing Voice Conversion](Singing%20Voice%20Conversion.md) model that uses [SoftVC](../../../permanent/softvc.md) content encoder to extract speech features from source audio.

The feature vectors are fed into [[VITS]] without the need for conversion to a text-based intermediate representation.

Therefore, the pitch and intonation of the original audio are preserved.

Meanwhile, the vocoder was replaced with [[NSF HiFiGAN]] to solve the problem of sound interruption.