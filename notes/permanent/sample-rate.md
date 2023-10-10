---
title: Sample Rate
date: 2023-10-08 00:00
modified: 2023-10-08 00:00
summary: A measure of how accurately the source signal was digitally represented
cover: /_media/sample-rate-cover.png
hide_cover_in_article: true
status: draft
---

The sample rate of audio indicates how frequently a [Sound Wave](sound-wave.md) amplitude is sampled to create its discrete digital representation. It's measured in samples per second or hertz.

Along with [Bit Depth](../journal/permanent/Bit%20Depth.md), sample rate is one of the key details that defines the quality of a digital audio recording.

If we zoom into an audio file in Audacity to the higher resolutions, we can visualise our waveform at the sample level:

![](../_media/sample-rate-1.png)

Each sample is simply represented as a number, of which the range will be based on the [Bit Depth](../journal/permanent/Bit%20Depth.md), of the audio file.

Therefore, digital audio is simply represented as an array of numbers. For stereo audio, it will be an array per channel. The number of elements in the array is $\text{ audio times (secs) } \times \text{ sample rate }$. Therefore a higher sample rate will require more storage space.

How do we determine the optimal sample rate?
## Nyquist-Shannon sampling theorem

According to the [Nyquist-Shannon Sampling Theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem), the frequency of sampling of a wave must be greater than twice the highest frequency in a wave. That's why a sample rate 44.1kHz was chosen as the standard for CD quality. 44.1kHz is a sensible default for high-quality audio, although some projects call for 48kHz.

However, there are other standard sample rates for different industries.
## Common Sample Rates

* 44.1 kHz - the most commonly found sample rate, as it has been the standard for CD quality since inception.
    * Based on the [Nuquist-Sannon](Nuquist-Sannon) sampling theorem, which states that the sample rate should be twice the highest frequency you want to reproduce. 
* 48 kHz - for film and video
* 88.2 kHz and 96 kHz -  for higher-resolution audio formats
* 192 kHz - for some ultra-high-definition recordings.

## Aliasing

Since the true sound wave has to be inferred from digital samples, the sound will only be accurately captured if the rate is higher. In particular, the higher frequencies will be folded into lower frequencies, causing distortion called aliasing.