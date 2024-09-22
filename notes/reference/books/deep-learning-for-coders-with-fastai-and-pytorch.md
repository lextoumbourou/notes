---
title: "Deep Learning for Coders with Fastai and Pytorch: AI Applications Without a PhD"
date: 2021-05-30 00:00
status: draft
---

## Chapter 1. Your Deep Learning Journey

* Deep learning is for everyone: doesn't require lots of maths, lots of data or lots of expensive compute, as people once had us believe.
* It's a computer technique to extract and transform data using multiple layers of neural networks.
    * Each layer takes input from previous layers and refines them.
    * Layers trained to minimise errors
* Neural networks history:
	* 1943
        * Mathematical model of artificial neuron is described in "A Logical Calculus of the Ideas Immanent in Nervous Activity" by Warren McCulloch (neurophysiologist) and Walter Pitts (logician)
            * Simplified model of a brain's neuron could be represented with addition and thresholding.
                * Thresholding example: `value = max(0, value)`
            * Walter Pitts was self-taught and homeless most of his life.
    * 1958
        * Work was later up by Frank Rosenblatt who developed the Mark 1 perceptron machine describe in “The Design of an Intelligent Automaton”
    * 1969
        * Perceptrons (MIT Press) by Marvin Minsky and Seymour Papert showed a single layer of neurons could not learn functions including XOR
            * However, same book also showed 2 layers could but that was larger ignored
    * 1986
        * Pivotal work in neural networks was release: multi-volume Parallel Distributed Processing (PDP) by David Rumelhart, James McClelland.
            * PDP approach very similar to today's approach with requirements:
                * A set of processing units
                * A state of activation
                * An output function for each unit
                * A pattern of connectivity among units
                * A propagation rule for propagating patterns of actvities through the network of connectivities
                * An activate rule for comping the inputs impinging on a unit with the current state of unit to product an output
                * A learning rule where patterns of connectivity is modified by experience
                * An environmnet within which system operates
	* Neural networks widely used in the 80s and 90s, but they weren't able to get very deep (adding more than 2 layers) without breaking or being too slow.
		* Only recently we've learned the power of neural networks with more compute, bigger datasets and some tweaks (like Dropout).
	* How to learn Deep Learning
		* Harvard professor David Perkins, who wrote Making Learning Whole (Jossey-Bass), teach the "whole game"
			* Baseball example: That means that if you’re teaching baseball, you first take people to a baseball game or get them to play it. You don’t teach them how to wind twine to make a baseball from scratch, the physics of a parabola, or the coefficient
			* In Deep Learning: helps if you have the motivation to fix model
		* Hardest part of DL is "artisanal": learning if your model is training properly, have you got enough data etc
	* Modern deep learning terminology
		* "Functional form" of model is architecture (people use model to mean architecture confusing)
		* Weights are parameters
		* Predictions are calculated from independant variables, data including the labels
		* Results of models are predictions
		* Measure of performance is loss
		* Loss depends on predictions but also correct labels

## Chapter 2

* [Out-of-Domain](../../permanent/out-of-domain-data.md) data (pg. 104)
    * Data that is given to a model that is different to the data it was trained on.
    * No complete technical solution: have to carefully roll out model
* [Domain Shift](../../permanent/domain-shift.md) (pg. 104)
    * Data that model performs inference on changes over time.
    * Example: types of customers an insurance sees changes over time, making their earlier models less useful.

## Chapter 5. Image Classifier

* The book is structured to teach all details of deep learning, but motivating each example with an actual problem.
* Datasets tends to be structured in one of 2 ways (pg. 213-214):
    * Files either in folders or with metadata (like labels) in the filename.
    * A CSV (or similar) where each row is an item. In image classification, each row may include a filename.
* L class (pg. 214-215)
    * fastai enhancement to Python's list.
* [Presizing](../../permanent/presizing.md) (pg. 216-219)
    * Images need to be same size to collate into tensors.
    * Wants to reduce operations for augmentations down to the minimum and do on the GPU.
    * If you're resizing before augmentations, it can degrade your image quality.
        * Eg a 45 degree rotation having empty space on 2 sides.
    * Steps:
        * 1. Get them uniform size by resizing image using random crop on largest axis to larger sizes than you're planning to use in model to get them uniform size.
                * On training set it's a random crop. On validation, it's a centre crop.
                * By doing this, you leave "spare margin (for) transforms on ... inner regions without creating empty zones"
        * 2. Combine all augmentation operations (include final resize) into one and do on GPU.
            * On validation, only resize is done.
* Train a simple model early (pg. 221)
* [Categorical Cross-Entropy Loss](../../permanent/categorical-cross-entropy-loss.md) (pg. 222-231)
    * Defined as: [Softmax Activation Function](../../permanent/softmax-activation-function.md) then [Negative Log-Likelihood](../../permanent/negative-log-likelihood.md)
    * [Exponential Function](../../permanent/exponential-function.md)
        * Defined as $e^x$
            * $e$ is a number about 2.718
                * The inverse of natural logarithm
        * Always positive and increases fast
    * [Softmax Activation Function](../../permanent/softmax-activation-function.md) (pg. 223-227)
        * The multi-category equivalent of the [Sigmoid Function](../../permanent/sigmoid-function.md)
            * Similarly "smooth and symmetric" properties
        * Use if more than 2 categories and want probabilities add to 1.
            * Can also use when there are just 2 categories to be consistent
        * If one of the numbers is slightly bigger, exponential amplifies it
        * Softmax "wants" to pick a single result.
            * Use model with multiple binary columns to handle items model hasn't seen.
        * One part of the [Categorical Cross-Entropy Loss](../../permanent/categorical-cross-entropy-loss.md)
    * Log Likelihood (pg. 226-231)
        * For a vector of softmaxed predictions, take the prediction that corresponds with the correct label.
        * Then apply `-log(prediction)`
            * In PyTorch, log uses $e$ as the base.
            * Note that because a log between 0 and 1, it has a negative log value. We invert it with negative.
            * The closer to 1, the closer to 0 loss.
        * In PyTorch, the `nll_loss` function doesn't take the log. It expects it to be already taken.
    * Consider gradient of `cross_entropy(a, b)` is `softmax(a)-b`
    * When `softmax(a)` is final activation, gradient is the same as diff between prediction and target
        * So it's the same a [Root mean-squared error - L2 Loss](../../permanent/Root mean-squared error - L2 Loss.md) in regression.
        * Because gradient is linear, don't see sudden jumps or exponential increases in gradients
* Model Interpretation
    * [Confusion Matrix](../../permanent/confusion-matrix.md) (pg. 232)
    * `most_confused` method for showing the items with highest loss (pg. 232-233)
* Improving the model
    * [Learning Rate Finder](Learning Rate Finder) (pg. 233-236)
        * Created by researcher Leslie Smith in 2015
        * Steps:
            * Train a model starting with a very small learning rate
            * Each batch, increase the learning rate by double (or some %)
            * Track the loss each step
            * When it doesn't get better, select learning rate order of magnitude less than min
    * Unfreezing and Transfer Learning (pg. 236-239)
        * CNN is many linear layers with non-linear activation function in between.
        * At the end is a last linear layer with a final activation like the [Softmax Activation Function](../../permanent/Softmax Activation Function.md).
        * In transfer learning, we start by replacing the last layer with one that has correct number of outputs for task.
        * Since we don't want to lose the learned weights in the earlier layers, we start by freezing those layers to just train the last layer.
    * Discriminative Learning Rates (pg. 239-241)
        * Earlier layers should in theory need less training, since they've learned abstract concepts like edge and gradients.
        * Set different (lower) learning rates for earlier layers.
        * In fastai, you can pass a [Python slice object](Python slice object) anywhere that accepts a single learning rate.
        * First value of slice is learning rate at start and last is final layer.
            * Layers in between have values that are evenly distanced between the 2 learning rates.
    * Selecting number of epochs (pg. 241-242)
        * When using 1cycle training, it's not advised to use the model that has the best accuracy in the middle of training, as the best values usually come when at the end when learning rate is lower.
        * If your model overfits, train model again with number of epochs equal to point of overfit.
    * Deeper Architectures (pg. 243-245)
        * Larger model (more layers) in theory should capture more complex relationships (though can also overfit).
            * Architectures like Resnet have small number of common varients: `18`, `34`, `50`, `101` simply because these happen to be the numbers that have pretrained models available.
        * Deeper architectures require more memory and therefore smaller batch sizes.
        * Bigger models aren't always better: start small and scale up.
        * Can use [mixed precision training](mixed precision training) to reduce memory size.
            * Use less-precise numbers (half-precision floating point, also called fp16) where possible during training.

## Chapter 6. Other Computer Vision Problems

* [Multi-label Classification](Multi-label Classification) *(pg. 248-249)*
    * Use when you dataset can include a number of true labels (or even none).
    * Can also be useful when you expect to see images that have none of your target classes.
        * A commonly reported problem with a simple solution that isn't widely applied.
* Book example uses the PASCAL dataset, which has more than one object per image.
* PyTorch and fastai classes for representing a dataset:
    * Dataset
        * Collection that returns tuple of independent and dependent varaible for a single item.
    * DataLoader
        * Iterator that gives you streams of mini-batches, where each mini-batch is a bunch of independent and dependent tuples.
    * Datasets
        * Iterator that contains a training Dataset and valid Dataset.
    * DataLoaders
        * Object that contains training and val DataLoader.
* When constructing a DataBlock, you can do a little bit at a time to ensure everything is working.
* [One hot-encoding](../../permanent/one-hot-encoding.md) (pg. 254-255)
    * A vector of 0s with a 1 in each location represented in data.
* [Binary Cross-Entropy Loss](../../permanent/binary-cross-entropy-loss.md) (pg. 256-257)
    * A multi-label classification loss function.
    * Nearly identical to Cross Entropy, except performs Sigmoid against each output logit.
    * Function:

            def binary_cross_entropy(inputs, targets)
                inputs = inputs.sigmoid()
                return -torch.where(targets==1, 1-inputs, inputs).log().mean()

    * In PyTorch:
        * `F.binary_cross_entropy` or `nn.BCELoss` can apply to logits that already have Sigmoid performed (equivalent to `F.nll_loss`)
        * Or `F.binary_cross_entropy_with_logits` and `nn.BCEWithLogitsLoss` which performs Sigmoid then binary cross entropy, equivalent to `nn.CrossEntropyLoss`.
* Accuracy with thresholding (pg. 259-261)
    * Normal accuracy compares the highest value prediction with the target (using `argmax`).
    * However, since there isn't one target, we can use thresholding to make a cutoff between a positive or negative prediction.

            def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
                "Compute accuracy when `inp` and `targ` are the same size."
                if sigmoid:
                    inp = inp.sigmoid()

                return ((inp>thresh)==targ.bool()).float().mean()
    * Finding the optimal threshold can be performed with a simple search over linear space on the validation set.
    * Some may fear overfitting, but if the curve appears smooth, it's likely something you can trust.
