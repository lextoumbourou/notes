---
title: See like a Radiologist with Systematic Windowing
date: 2019-10-19 00:00
link: https://www.kaggle.com/jhoward/don-t-see-like-a-radiologist-fastai
status: draft
category: reference/kaggle/rsna-intracranial-hemorrhage-detection
---

* radiologists use windowing to increase the contrast of images across bands of Hounsfield Units
* They need to do this because human visual system can only see 100 levels of gradation of a single colour (white/grey/black) and there at 2^17 levels in a a DICOM image.
* You could squish down in to 256 levels, or use different windows to make a 3-channel image, but you still have to throw away some data.
* Instead, since a neural network takes floating point data, you can just use the values directly.
* However, can't use PIL which operates on 8-bit data and can't use JPEG, which also only supports 8-bit data.
* Rescaling floating point data
	* Can't ignore scaling completely.
	* having well-scaled inputs is really important to getting good results from your neural net training
	* That means we want to see a good mix of values throughout the range of our data - e.g. something having approximately a normal or uniform distribution. Unfortunately, the pixels in our DICOM above don't show that at all:

Missing stuff here!

* Afterword: a better windowing for humans?
	* Can use a rainbow colour map to fully utilise computer's ability to display colour in radiologist imagery
