# Week 6

* Linear classifiers:

  * Creates a linear decision boundary between classifications.

* Neural networks

  * Represent classifiers using graphs.
    * Allows for complex, non-linear features.

  <img src="./basic-neural-network.png"></img>

  * ``X[]`` - represents features.
  * ``W[]`` - represents weights.
  * ``W0`` - gets multiplied by ``1`` (for some reason?).

  * Useful for cases which linear classifiers can't find.

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
