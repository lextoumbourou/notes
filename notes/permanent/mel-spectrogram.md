---
title: Mel Spectrogram
date: 2023-10-09 00:00
modified: 2023-10-09 00:00
cover: /_media/melspectrogram-cover.png
summary: A sound wave visualisation of frequency over time
tags:
  - AudioEngineering
hide_cover_in_article: true
---

Mel Spectrogram is a graphic representation of a [Sound Wave](sound-wave.md), visualising frequency over time.

Here is a Mel Spectrogram of the audio of a Trumpet

![Melspectrogram example of a Trumpet](../_media/melspectrogram-example.png)

<audio controls>
  <source src="/_media/trumpet_example.mp3" type="audio/mpeg">
</audio>

The process of generating a Mel Spectrogram works like this:

1. Break the audio signal down into short frames
2. Convert time signal into the frequency domain using a [Fourier Transform](fourier-transform.md)
3. Convert frequencies into Mel scale, to more closely align with our intuition of frequencies. This operation is called [Mel Filter Bank](mel-filter-bank.md).
4. Plot the Mel values over time.

## Mel Scale

The Mel Scale is a *perceptual scale* of audio frequencies. In other words, it represents our perceived distance of the frequencies from others.

The Mel scale is a logarithmic formula where 1000Mel = 1kHz. You can convert Hz to Mel using this formula:

$Mel(f) = 2595 \log_{10} (1 + \frac{f}{100})$

![Frequency vs Mel Scale plot](../_media/mel-scale-plot.png)
