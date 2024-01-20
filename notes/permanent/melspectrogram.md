---
title: Mel Spectrogram
date: 2023-10-09 00:00
modified: 2023-10-09 00:00
cover: /_media/melspectrogram-cover.png
summary: A visualisation of pitch energy over time for a sound wave
tags:
  - AudioEngineering
---

Mel Spectrogram is a graphic representation of a [Sound Wave](sound-wave.md), visualising pitch energy over time.

![Melspectrogram](../_media/melspectrogram-example.png)
<audio controls>
  <source src="/_media/trumpet_example.mp3" type="audio/mpeg">
</audio>

How they're generated:

1. Break the audio signal down into short frames
2. Use a [Fourier Transform](../../../permanent/Fourier%20Transform.md) to convert the time signal into the frequency domain.
3. Convert the frequency domain into the a [Mel Scale](../../../permanent/mel-scale.md) using a [Mel Filter Bank](../../../permanent/mel-filter-bank.md), which extracts energy information at each frequency band.

## Mel Scale

The Mel Scale is a *perceptual scale* of audio frequencies, which is a scale of audio based on our perceived distance from each other.

The Mel scale is a logarithmic formula where 1000Mel = 1kHz. You can convert Hz to Mel using this formula:

$Mel(f) = 2595 \log_{10} (1 + \frac{f}{100})$

![Mel Scale](../_media/mel-scale-plot.png)