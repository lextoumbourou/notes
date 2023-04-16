---
title: "fastai - Lesson 1 - Deep Learning for Coders (2020)"
date: 2021-06-10 00:00
category: reference/moocs
cover: /_media/fastai-2020-lesson-1-cover.png
summary: "Notes taken from the Deep Learning for Coders (2020) - Lesson 1 video"
parent: fast-ai-2020
status: draft
---

Notes taken from watching the [Lesson 1 - Deep Learning for Coders (2020)](https://www.youtube.com/watch?v=_QUEXsHfsA0) video.

## 00:00 - Intro

* Course is "definitive version" after 4 years of Fastai
    * Includes book and peer-reviewed paper on library.
* Syllabus of course based on the book

## 03:34 - Course format

* Book was written in notebook
    * Course includes version of course without prose and output to run while learning
* Coauthors of course:
    * Sylvain Gugger coauthor of book
    * Rachael Thomas cofounder of fastai

## 06:31 - Misconceptions about requirements for Deep Learning

* What you don't need for DL:
    * Don't need a lot of math: high school level is enough
    * Don't need a lot of data: world class results can be obtained with < 50 examples
    * Don't need expensive computers: can train state of the art models for free
* Course requires coding ability, preferably in Python (though can be picked up during the course)

## 08:38 - Deep learning best known approach in a number of areas

* Image classification, image segmentation, NLP, some structured data problems
* In some areas, equal or better than human performance

## 09:54 - Neural networks history

* Note: Deep Learning a type of neural network learning (a deep one)
* In 1943, neurophysiologist, Warren McCulloch, and logician, Walter Pitts created model of artificial neuron
* Declared that since brain activity had an "all-or-nothing" characteristic, they could be treated as "propositional logic"
* In 1950s, Frank Rosenblatt made some subtle changes to model and oversaw the creation of the [Mark 1 Perceptron](permanent/mark-1-perceptron.md)

    ![Mark 1 Perceptron](/_media/mark-1-perceptron.jpeg)

## 11:22 - First AI winter

* Marvin Minsky created book called Perceptrons about that invention
* Showed that a single layer could not learn critical function like XOR
    * Paper showed that multiple layers could, however, people ignored that part of the book

## 12:14 Parallel Distributed Processing (PDP)

* Released in 1986, described the foundations of modern deep learning:

    ![PDP](/_media/pdp.png)

## 13:38 The Age of Deep Learning

* 2nd layer of neurons added
    * Shown that extra layer of neurons could approximate any function
        * In practice, more layers are usually required

## 16:00 Top-down learning

* Unlike other ML courses, doesn't start with refresher on calculus, or lessons on Sigmoid function
* Approached based on [Teaching the whole game](../../../../permanent/teaching-the-whole-game.md) by Professor David Perkins and others from Harvard.

## 23:09 Software stack

* Software used:
    * fastai library, which is built on PyTorch built on Python
* Vast majority of Deep Learning researchers using PyTorch: industry still Tensorflow
* PyTorch not designed for beginner friendlyness
* fastai designed to be flexible but also useful for learning and teaching

## 27:36 GPUs

* GPU: Graphics Processing Unit
    * Requires a Nvidia GPU: other brands aren't supported by Deep Learning libraries.
* Use existing platform setup for it: not worth your time building your own machine while learning

## 31:30 Juypter Notebooks overview

* Juypter Notebook is REPL with powerful features
* Course provides a notebook that shows Jupyter functionality
* If not familiar with notebooks, can be as awkward as moving from gui to command-line
* Has 2 modes: command-mode and edit-mode
* Can format text in Markdown
* Can access terminal from Jupyter

## 39:08 Repositories overview

* fastbook repo - prose for book
* coursev4 repo - notebooks with all book-related stuff removed

## 40:32 Questionaires

* Each lesson has a questionaire
* Lots of effort put into questions to help answer the question: have I learned everything before moving on
* If you get stuck on a single question, move on and consider coming back

## 42:20 Running the first notebook

{% notebook reference/moocs/fast.ai/dl-2020/lesson-1-1.ipynb %}

## 48:00 Machine Learning

* Deep learning is a kind of Machine Learning
* Without Machine Learning, it would be near impossible to write a program to classify cats and dogs

## 49:05 Machine Learning history

* In 1949, Arthur Samuel started looking at alternative methods of completing task
    * Popularised the term: "Machine Learning"
* In 1962, wrote essay: "Artificial Intelligence: A Frontier of Automation" in it he describes the problem ML solves:
    * "Programming a computer for such computations is, at best, a difficult task, not primarily because of any inherent complexity in the comptuer itself but, rather, because of the need to spell out every minute step of the process in the most exasperating detail. Computers, as any programmer will tell you, are giant morons, not giant brains."
* Idea was that instead of telling computer how to solve a problem, give computer examples of problem and have it figure it out
    * Create a weight assigment to test effectiveness of problem, and provide mechanism of altering weights to maximise performance
* After training "model" you have something like: inputs -> model -> results
* When we use a model to do a task like playing checkers, we call it [Inference](../../../../permanent/inference.md)
    * Jargon: Machine Learning
        * Training of programs developed by allowing computer to learn from experience, rather than manually coding the steps

 ## 53:45 Neural networks & image classification

 * Is there is a function so flexibilty that the weights can do anything? Yes, a neural network.
 * Mathematical proof called [Universal Approximation Theory](permanent/universal-approximation-theory.md)
     * Function can solve any problem to any level of accuracy if you just find the right set of weights (in theory)
* [Stochasic Gradient Descent](../../../../permanent/stochasic-gradient-descent.md)
    * The "mechanism of altering weights to maximise performance" that Arthur Samuel referred to.
    * Course will look at exactly how it works.
* Terminology used nowadays as compared to Samuel's:
    * Functional form of model is [Architecture](Architecture)
    * Weights are called [Parameters](Parameters)
    * Predictions calculated from [Independant Variable](Independant Variable), which is the data that doesn't include the labels
    * Results of model are called [Predictions](Predictions)
    * Loss depends on having [Predictions](Predictions) and correct [Labels](Labels)

## 58:57 Limitations of Machine Learning

* Model can't be created without data.
* Model can only learn based on patterns in input data
* Learning creates predictions *not recommended actions*
* Model can only learn on patterns in input data used to train
* Model requires labels to learn
    * Labels can be difficulty to get lots of dependant variables

## 01:01:32 Model operates in environment

* Consider the "environment" part of the PDP model
    * Feedback loop of model can create vicious circle.
    * Example:
        * predictive policing model predicts where arrests are likely to occur.
        * more police therefore go to the area, which leads to more arrests.
        * that is fed into model, which creates more arrests.
    * Consider that predicting arrests is an example of where [Metrics Are Proxies](permanent/Metrics Are Proxies.md) for what you care about: reducing crime.
        * Sometimes the difference between proxy and what you are actually trying to do can be significant

## 01:04:14 Dogs and Cats Notebook review

### 01:04:30 Use of import *

* In Python, you have to import everything you need.
* Python provides a convenience tool called `import *`
    * Usually not recommended as it adds a whole bunch of stuff to your namespace
* Fastai is designed for Repls, therefore, thought has gone into making `import *` safe
    * Question: why is it so hard to import what you need in a Repl?
* Can view location of symbol in Repl by putting symbol in a cell and pressing shift + enter

### 01:09:05 review continued

* `doc` function: Used to show documentation for a method or function in fastai
* `untar_data`: Downloads dataset, decompresses and puts on your computer
* `label_func`: Used to extract label from file
* `resnet34`: good starting point model for image classification
* `valid_pct`: sets part of the dataside aside (size determined by %) for validation to check you aren't overfitting

## 01:11:56 Overfitting

* A function can particular data really well, but that doesn't mean it will generalise to new data.

![Proper vs Overfit](/_media/proper-vs-overfit.png)

* "Craft of Deep Learning is about creating models with proper fit"
* Only way to do it is to check the metrics on data the model wasn't trained on.

## 01:13:50 fastai will improve your programming skills

* May show you a style of coding (especially in Python) you're unfamiliar with
* Lots of times fastai doesn't follow traditional approach to Python
    * If you use Python at work, likely want to follow a different style guide to fastai's

## 01:06:23 Fastai applications

* The same API used for classify Cats and Dogs can be used for:
    * Segmentation (classifying pixels)
    * Natural language processing (for example, sentiment analysis)
    * Structured data
    * Collaborative filtering
