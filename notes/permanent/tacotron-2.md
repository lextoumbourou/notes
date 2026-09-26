---
title: Tacotron 2
date: 2026-09-26 08:52
modified: 2026-09-26 08:52
status: draft
---

**Tacotron 2** is a neural text-to-speech system from Google, described in the paper [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](natural-tts-synthesis-by-conditioning-wavenet-on-mel-spectrogram-predictions.md).

It has two parts:

1. A recurrent sequence-to-sequence network with attention, which predicts a [Mel Spectrogram](mel-spectrogram.md) from the input characters.
2. A modified [Wavenet](wavenet.md) vocoder, which generates the audio waveform from the predicted mel spectrogram.

Using the mel spectrogram as the intermediate representation meant the system could be trained directly from text and audio pairs, without the hand-engineered linguistic features earlier systems relied on.
