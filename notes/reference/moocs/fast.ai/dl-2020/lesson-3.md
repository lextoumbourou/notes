---
title: "fastai - Lesson 3 - Deep Learning for Coders (2020)"
date: 2021-06-27 00:00
category: reference/moocs
cover: /_media/fastai-2020-lesson-3-cover.png
summary: "Notes taken from the Deep Learning for Coders (2020) - Lesson 3 video"
parent: fast-ai-2020
status: draft
---

Notes taken from watching the [Lesson 3 - Deep Learning for Coders (2020)](https://www.youtube.com/watch?v=5L3Ao5KuCC4) video.

## 00:00:00 - Recap lesson 2

* Last lesson about getting model into production
* Today's lesson:
    * Learn foundations of what happens when training a neural network
    * Learn about [Stochastic Gradient Descent](../../../../permanent/stochastic-gradient-descent.md)
* Order different from book. This lesson is Chapter 4 and will do Chapter 3 later.

## 00:01:10 - Resizing images with Datablock

* Images in a batch should be converted to same size.
    * Making them squares is most common approach.
    * To resize each image to 128x128: `item_tfms=Resize(128)`
        * By default it crops the largest side and resizes.
 * To make changes to a Datablock, without affecting the original use `Datablock.new()`
     * Example below changing resize method from `Crop` to `Squish`:

{% notebook reference/moocs/fast.ai/dl-2020/lesson-3-1.ipynb %}

## 00:33:40 - GPUs or CPUs in production

* In production, since you're usually doing one image at a time, a CPU is likely more efficient for inference.
    * Deploying to a CPU is also cheaper and more readily available than GPUs
* When to use a GPU in production?
  * Processing videos or audio
  * Huge volumes per second, where requests can be batched
      * However, if volume is too low users may be waiting

## 00:36:12 - Deploying to a phone

* Recommendation: deploy to a server, then have phone talk to server
* Trying to get PyTorch to run on phone will require you to convert it to other format using [ONNX](https://onnx.ai/)
    * Hard to debug and maintain

## 00:38:17 - Avoiding disaster

* 2 different types of deployment: hobby apps / prototypes and real-life products
* Things to be careful of:
    * Models can only reflect data used to train them
        * Healthy skin example from [Actionable Auditing: Investigating the Impact of Publicly Naming Biased Performance Results of Commecial AI Products](https://www.media.mit.edu/publications/actionable-auditing-investigating-the-impact-of-publicly-naming-biased-performance-results-of-commercial-ai-products/) by Deb Raji:
            * When you search for "healthy skin" it returns results of young, white woman touching their face.
                ![Healthy skin example](journal/_media/healthy-skin-example.png)
* Think carefully about types of data you'll see in practice, and make sure it's in the source data.
* If nothing else, make sure test set reflects the data your model will see
* Lots of issues you'll see when deploying models covered in [Building Machine Learning Powered Applications](https://www.amazon.com/Building-Machine-Learning-Powered-Applications/dp/149204511X) by Emmanual Ameisen.

## 00:42:33 - Out-of-domain data and domain shift

* Consider difference between bear detector in real-life using cameras in a national park vs the prototype:
    * Videos not images
    * Low-res images
    * Bears will be in positions no one would bother uploading to the net
    * Would need to be really fast returning results
    * Extra: would have to be really careful about [Type 2 Errors (False Negatives)](permanent/type-2-errors.md).
* [Out-of-Domain](../../../../permanent/out-of-domain-data.md) data
    * Data that is given to a model that is different to the data it was trained on.
* [Domain Shift](../../../../permanent/domain-shift.md)
    * Data that you models see changes over time.
    * Example: insurance company using deep learning model for pricing and risk.
        * Over time customers they serve change, and so would the risk profile
        * Model's trained on older data are no longer relevant.

## 00:44:58 - All data is bias

* All data is biased: there's no perfectly representative data
* Proposals to address it, like [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) suggest writing down attributes of the dataset:
    * Understand how data was gathered
        * How was it collected?
        * Who collected it?
    * What are its limitations?

## 00:45:55 - High-level approach to mitigate risks in ML products

1. Start out doing it manually and have model running alongside person
    * Continually compare human predictions to model
2. Deploy with a limited scope
    * Ensure you have processes to supervise it
    * Limit by time and scope
3. Gradual expansion
    * Have good reporting/monitoring systems in place
    * Consider what can go wrong

## 00:49:51 - Question: does fastai have tools for incrementing improving models?

* You don't need anything special in fastai - just include data from production and retrain.

## 00:51:00 - Feedback loops and unforeseen consequences

* When you roll out model, it might change the behaviour of system it's a part of
    * Could take a minor issue and "explode" into a big issue.
    * Predictive policing example:
        * Predicts crime in certain areas, causing police to go there
        * More police means more arrest, which in turns makes model more confident in predictions
        * An example of where [Proxy Metrics](../../../../permanent/proxy-metrics.md) - you want to reduce crime, but use arrests as a proxy for crime
    * Issue is particularly a problem in the prescence of bias
    * Risk of having a feedback loop is anytime your model controls what your future data looks like
    * Consider human's in the loop: appeals process, monitoring etc

## 00:57:22 - Writing and blogging

* Think about starting writing before you go much future.
    * Keeping on top of what you're learning by writing.
* [Fastpages](https://github.com/fastai/fastpages) is a platform for blogging with Juypter notebooks
* "Write for yourself 6 months ago"

## 01:04:09 - MNIST from scratch

* Course now moves into lower-level details of Deep Learning, until next lesson on ethics.

{% notebook reference/moocs/fast.ai/dl-2020/lesson-3-2-mnist.ipynb %}
