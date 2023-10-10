---
title: Sample Rate
date: 2023-10-08 00:00
modified: 2023-10-08 00:00
summary: A measure of how accurately the source signal was digitally represented
cover: /_media/sample-rate-cover.png
hide_cover_in_article: true
---

The sample rate of audio indicates how frequently a [Sound Wave](sound-wave.md)'s amplitude is sampled to create a digital representation. It's measured in samples per second or [hertz](https://en.wikipedia.org/wiki/Hertz).

Along with [Bit Depth](../journal/permanent/Bit%20Depth.md), the sample rate is one of the critical details that define the quality of a digital audio recording.

If we zoom into an audio file in Audacity to the higher resolutions, we can visualise our waveform at the sample level:

![](../_media/sample-rate-1.png)

Each sample is represented as a number, either an integer or float; the range will define the [Bit Depth](../journal/permanent/Bit%20Depth.md), of the audio file.

Therefore, digital audio is represented as an array of numbers. The number of elements in the collection is $\text{ audio time (secs) } \times \text{ sample rate }$. Therefore, a higher sample rate will require more storage space. For stereo audio, it will be an array per channel.

In Python, the sound is typically represented using a Numpy multidimensional array. Here we can see an example of loading an audio file using the [scipy](https://scipy.org/) library:

```python
>>> from scipy.io import wavfile
>>> sample_rate, audio_array = wavfile.read("../../_media/4s-silence.wav")
>>> audio_array.shape
(176400,)
>>> audio_length = len(audio_array) / sample_rate
>>> audio_length
4.0
```

As you can see, we can find the length of audio represented as a Numpy array by dividing the number of samples by the sample rate.

How do we determine the optimal sample rate?

## Nyquist-Shannon sampling theorem

According to the [Nyquist-Shannon Sampling Theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem), the frequency of sampling of a wave must be greater than twice the highest frequency in a wave. That's why a sample rate 44.1kHz was chosen as the standard for CD quality. 44.1kHz continues to be a sensible default for high-quality audio, although 48kHz is also a common choice.

However, there are other standard sample rates for different types of audio.

## Common Sample Rates

* 44.1 kHz - the most commonly found sample rate, as it has been the standard for CD quality since inception.
* 48 kHz - for audio and some film and video
* 88.2 kHz and 96 kHz - for higher-resolution audio formats
* 192 kHz - for some ultra-high-definition recordings.

## Aliasing

Since the true sound wave has to be inferred from digital samples, the sound will only be accurately captured if the rate is higher. In particular, the higher frequencies will be folded into lower frequencies, causing distortion called aliasing.

The figure below shows an example of a 15Hz sine wave over a minute. As you can see, we cannot accurately reconstruct the original sine wave if we do not sample enough points. However, after a certain number of samples, we can rebuild the sound wave perfectly; more samples do not help.

![](../_media/sample-rate-examples.png)

Real sound waves are more complex than simple sine waves, so more samples are needed to capture that complexity. However, the important detail is that more samples are not necessarily better.