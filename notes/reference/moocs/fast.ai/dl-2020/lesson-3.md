---
title: "fastai - Lesson 3 - Deep Learning for Coders (2020)"
date: 2021-06-19 00:00
category: reference/moocs
tags:
  - DeepLearning
summary: "Notes taken from the Deep Learning for Coders (2020) - Lesson 3 video"
status: draft
---

## 00:00:00 - Recap lesson 2

* Last lesson at getting model into production
* Today's lesson:
    * Learn foundations of what is happening when training neural network
    * Learn about SGD
* Order different from book. This lesson is Chapter 4 and will do Chapter 3 later.

## 00:01:10 - Resizing images with Datablock

* For each batch, need to convert all into same size. Making them squares is most common approach
* `item_tfms=Resize(128)` - each image in a batch will be resized to 128x128 by Cropping them
* Another way to do it is to make a Datablock object from another Datablock object and change just some pieces. Here I can change resize algorithm from crop to squish:

{% notebook reference/moocs/fast.ai/dl-2020/lesson-3-1.ipynb %}

## 00:33:40 - GPUs or CPUs in production

* A CPU is actually more efficient than a GPU, since you are usually doing one item at time.
* Deploying to a CPU is cheaper and easier than deploying to a GPU
    * Allows you to use any provider you like.
* When to use a GPU?
  * Processing videos or audio
  * Huge volumes per second, where requests can be batched. 
      * However, if volume is too low users may be waiting

## 00:36:12 - Deploying to a phone

* Recommendation: deploy to a server, then have phone talk to server
* Trying to get PyTorch to run on phone will require you to convert it to other format, like ONNX
    * Hard to debug and maintain

## 00:38:17 - Avoiding disaster

* 2 different types of deployment: hobby apps / prototypes and real-life products
* Things to be careful of:
    * Models can only reflect data used to train them
        * Healthy skin example: when you search for healthy skin it returns results of young, white woman touching their face
        
            ![[Pasted image 20210620154907.png]]
            
* Think carefully about types of data you'll see in practice, and make sure it's in the source data
* Example from paper by Deb Raji [Actionable Auditing: Investigating the Impact of Publicly Naming Biased Performance Results of Commecial AI Products](https://www.media.mit.edu/publications/actionable-auditing-investigating-the-impact-of-publicly-naming-biased-performance-results-of-commercial-ai-products/)
* Test set in particular should reflect real world examples that your model will see in production
* Lots of issues you'll see when deploying models
    * Read book [Building Machine Learning Powered Applications](https://www.amazon.com/Building-Machine-Learning-Powered-Applications/dp/149204511X) by Emmanual Ameisen which covers many of them.

## 00:42:33 - Out of domain data and domain shift

* Consider a bear example in practice:
    * Deployed to detect bears in a national park using surveillance cameras
        * Using videos instead of images
        * Low resolution camera images
        * May see bears in positions rarely uploaded to the internet
        * May need results returned extremely fast
* [[Out-of-domain data]]
    * Data that is given to a model that is different to the data it was trained on.
* [[Domain shift]]
    * Data that you models see changes over time.
    * Insurance company may use deep learning model for pricing and risk algorithm, but over time types of customers they have may change, making original training data no longer relevant

## 00:44:58 - All data is bias

* All data is bias: there's no perfectly representative data
* Proposals to address it, like [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) suggest writing down attributes of the dataset, how it was collected, who collected it etc are to make you aware of the sort of limitations of a dataset, avoid being blindsided later
    * Understand how data was gathered
    * What are its limitations?

## 00:45:55 - High-level approach to mitigate risks with ML products

* Start out doing it manually and have model running alongside process
    * Compare human predictions to model
* Deploy with a limited scope
    * Ensure you have processes to supervise it
    * Limit by time and scope
* Gradual expansion
    * Have good reporting/monitoring systems in place
    * Consider what can go wrong

## 00:49:51 - Question: does fastai have tools for incrementing improving models?

* You don't need anything special - can just include data from production and retrain

## 00:51:00 - Feedback loops and unforeseen consequences

* When you roll out model, it might change the behaviour of system it's a part of
* Take a minor issue and "explode" into a big issue.
* Predictive policing example:
    * Predicts crime in certain areas, causing police to go there
    * More police means more arrest, which in turns makes model more confident in predictions
    * An example of where [[Metrics Are Proxies]] - you want to reduce crime, but use arrests as a proxy for crime
* Issue is particularly a problem in the prescence of bias
* Risk of having a feedback loop is anytime your model controls what your future data looks like
* Consider human's in the loop: appeals process, monitoring etc

## 00:57:22 - Writing and blogging

* Think about starting writing before you go much future.
    * Keeping on top of what you're learning by writing.
* [Fastpages](https://github.com/fastai/fastpages) is a platform for blogging with Juypter notebooks
* Write for yourself 6 months ago

## 01:04:09 - MNIST from scratch

* Course now moves into lower level details of Deep Learning, until next lesson on ethics.
* MNIST
    * Handwritten digits
    * Collated into ML dataset by Yann Lecun
* Start with something really simple and scale it up

{% notebook reference/moocs/fast.ai/dl-2020/lesson-3-2-mnist.ipynb %}