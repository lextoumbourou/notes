---
title: Structuring Machine Learning Projects (Coursera) - Week 2
date: 2017-09-30 00:00
link: 
category: reference/moocs
status: draft
parent: machine-learning-projects
---

## Error analysis

### Carrying out error analysis

* Manual examining errors your classifier is making, can help you figure out what to do next.
* Question: should you try to make your cat classifier do better on dogs?
* Error analysis procedure:
    * Get ~100 mislabled dev set examples.
    * Count are how many are dogs.
    * If it's 5%, you could improve error from 10% to 9.5%
    * May not be best use of your time, but at least provides a "ceiling on performance".
    * If it's 50%, then you can be sure this is the best use of your time.
* Can evaluate ideas in parallel:
    * Create a spreadsheet with different examples and why they were miscategorised to start to figure out patterns and where you get the best bang for buck with improvements.

### Cleaning up incorrectly labeled data

* Deep learning algorithms can handle some random errors in the training dataset, if that dataset size is big enough.
* Deep learning is less robust to systematic errors.
* With incorrectly labeled examples in dev/test set, can add an extra column to the error analysis spreadsheet described previously with incorrectly labelled examples.
  * If the % is high enough, maybe it's worth your time to fix.
  * Want to look at 3 examples to decide:
      1. Overall dev set error.
      2. Errors due to incorrect labels.
      3. Errors due to other causes.
* If cleaning up dev set, want to make sure you clean both dev and test set.
* Consider examples your algo got right, as well as wrong.

### Build your first system quickly and iterate from there

* Having some trained system lets you start analysing things like bias/variance.

## Mismatched training and dev/test sets

### Training and testing on different distributions

* People want to have as much data as possible in the training set, and will to use data from a different source or "distribution" than the dev/test set.
* Example: expected data is blurry phone pics but you don't have enough for a training set, so you find high quality online.
    * Option 1: mix high quality with blurry and shuffle, then split to training/dev/test.
        * Problem: won't get enough of the expected images in the dev/test set.
    * Option 2: put the high quality images in the training set, with maybe some blurry images and ensure dev / test set has all blurry images.
* Speech recognition example: building a speech recognition mirror.
      * Training data might come from a big archive of purchase data and collected data
      * Dev/test comes from recordings from the rear view mirror.

### Bias and variance with mismatched data distributions

* Can be difficult to evaluate your training performance when training and dev are from different distributions.
  * Add an extra set called training-dev set from the training distribution.
* Called a "data mismatch" when performance is good on training-dev but not dev.

  ![Data mismatch](/_media/data-mismatch.png)

* No "super systematic" ways to go about addressing data mismatch.

### Addressing data mismatched

* Carry out manual analysis and try to understand difference between training and dev set (should ignore the test set).
    * Dev set may be noisy - try to augment dataset with noisy data.
* Artificial data synthesis
    * Add car noise to the background of audio recordings, if it matches the dev set.
    * Car image recognition: could you generate a training set from computer graphics?
		* Likely to overfit on the computer graphics.

## Learning from multiple tasks

### Transfer Learning

* Transferring learning from one model to another domain.
    * Remove last layer of model and retrain with another dataset.
    * Learnings from another dataset, could help speed up your dataset (already knows about lines, curves, objects etc).
* Options for retraining models:
  * If you have a small dataset, just retrain last layer (or last two).
      * Aka finetuning.
  * Lots of data? Could retrain all layers.
      * Aka pretraining.
* Could add additional layers of the network for certain speech recognition tasks.
* Transfer learning makes sense if:
    * Task A and B have the same inputs.
    * You have a lot more data for A than B.
    * Low level features of A (shapes, features etc) could be helpful for Task B.

### Multitask learning

* The process of assigning multiple output labels instead of a single from multiple classes (like in Softmax regression).
* Loss function takes into account the loss from each output label:
  $\frac{1}{m} \sum\limits_{i=1}^{m} \sum\limits_{j=1}^{4} L(\hat{y}_i^{(i)}, y_j^{(i)})$
* Could also train 4 separate neural networks, but when the output classes can share information, you tend to get better performance with 1 network.
* Multitask learning makes sense if:
    * Training on a set of tasks that could benefit from having lower-level features.
    * Amount of data you have for each task is quite similar.
    * Can train big enough network to do well on all tasks.
* Only time multi-task learning may be bad for perform? If you can't train a big enough network (according to Rich Carona).

## End-to-end deep learning

### What is end-to-end deep learning?

* Take multiple stages of a data pipeline and replace with a single neural network.
* Speech recognition example: might have a pipeline that takes audio and extracts features (phonemes etc), then another that finds words and returns transcript.
	* End-to-end: take audio, return transcript.
* Face recognition at turnstile:
	* Found it better to detect face first, then crop the face out to feed into face classifier. Example where multiple neural nets works better than end-to-end analysis.
* Machine translation example:
	* With big enough dataset, end-to-end seems to work well. Smaller datasets may require some advanced feature engineering techniques.

### When to use end-to-end and when not to use

* Pros:
	* Let the data speak for itself.
	* Less hand designing of features.
* Cons:
	* Need very large dataset.
	* Excludes potentially useful hand-designed components.
* Key question: do you have enough data to learn the function of the complexity needed to map x to y?
