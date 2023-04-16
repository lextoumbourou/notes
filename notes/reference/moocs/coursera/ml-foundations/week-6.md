---
title: "Week 6 - Deep Learning: Searching for Images"
date: 2015-10-07 00:00
category: reference/moocs
status: draft
parent: ml-foundations
---

* Neural networks
    * Represent classifiers using graphs.
        * Allows for complex, non-linear features.

            ![Basic neural networks](/_media/ml-foundations-basic-neural-network.png)

    * ``X[]`` - represents features.
    * ``W[]`` - represents weights.
    * ``W0`` - not multiplied by any feature, gets multiplied by ``1`` (I guess it's a kinda "hard coded" part of the equation).
* One layer neural network:
    * Basically the same as a linear classifier:
        * Creates a linear decision boundary between classifications.
        * Can represent: ``X[1] OR X[2]``, ``X[1] AND X[2]``
        * Can't represent XOR functions.
* XOR problem can be solved by adding a layer.
    * [Video missing...](https://www.coursera.org/learn/ml-foundations/lecture/iJyru/learning-very-non-linear-features-with-neural-networks/discussions/Qm4R4G9OEeWDzg4yGnIlTw#input-container) guess I'll never know...
* Application of deep learning to computer vision
    * Image features
        * Features == local detectors
        * Nose detector, eye detector
        * Combined to make prediction.
        * In reality, features are more low-level.
    * Collections of locally interesting points
        * Combined to build classifiers
    * Standard approach (without neural networks):
        * Basically had to build by hand.
        * Extract features from input (similar to extracting features from documents).
        * Feed to simple classifier (logistic regression, SVMs).
    * Deep learning: implicity earns features
        * Multiple layers.
        * Different features are captured at different levels.
* Deep learning performance
    * [SuperVision winning ImageNet in 2012](http://www.technologyreview.com/view/530561/the-revolutionary-technique-that-quietly-changed-machine-vision-forever/) was a turning point in computer image recognition.
* Other examples of deep learning in computer vision
    * "Scene parsing" - figuring out what stuff is in a picture.
    * Retrieving similar images: take input image and output nearest neighbour.
* Pros of Neural Networks
    * Enables learning of features rather than hand tuning.
    * Impressive performance gains
        * Computer vision
        * Speech recognition
        * Some text analysis
    * Potential for more impact
* Cons
    * Requires *lots* of high-quality, labeled data
    * Lots of complexity
    * Computationally expensive
    * Hard to tune
* Deep features
    * Let's you build neural networks without a lot of data.
    * "Transfer learning"
        * Use data from one task to learn another
        * Take features from experiment to aid another.
    * Use the layers in neural net that make sense for dataset, and use simple classifier for the layers that do not.
        * Eg Layer 1 and Layer 2 for an algorithm for detecting butterflies could be use for algorithm to detect moths.
