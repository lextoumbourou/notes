---
title: melspectrogram
date: 2023-10-09 00:00
modified: 2023-10-09 00:00
status: draft
---

A visualisation technique for [[Sound Waves]].

Humans Perceive Frequency Logarthimically

Ideal audio feature representation:
* Time-frequency representation.
* Perceptually-relevant amplitude representation.
* Perceptually-relevant frequency representation.

Mel spectrograms
* Mel
    * Melscale - percuptually informed scale for pitch
![](../journal/_media/melspectrogram-melscale.png)
* Equal distance on the scale, have the same "perceuptual distance"
* 1000 Hz = 1000 Mel.
* Arrived at it from imperical experiments. Nothing magical.

$m = 2595 \cdot \log(1. +\frac{f}{500}$
$f = 700(10^{m/2595}-1)$

* How to extract:
    * Extract STFT
    * Convery amplitue to DBs
    * Convert frequnect to Mel scale
        * 1. Choose number of mel bands
        * Construct mel filter banks
        * Apply mel filter banks to spectro gram

* How many mel bands?
    * Depends on the problem.
