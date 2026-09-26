---
title: Spectrogram
date: 2025-04-09 00:00
modified: 2026-09-26 08:55
status: draft
---

A **Spectrogram** is a 2D plot of time vs frequency for a 1D signal, where the colour (or brightness) at each point shows how much energy there is at that frequency at that time.

It's usually computed by taking the [Fourier Transform](fourier-transform.md) of short overlapping windows of the signal (the Short-Time Fourier Transform). Where a [Spectrum](spectrum.md) shows the frequency content at one point in time or across the whole signal, a spectrogram shows how it changes over time.

See also [Mel Spectrogram](mel-spectrogram.md).
