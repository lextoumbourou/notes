---
title: "Week 3 - Classification: Analyzing Sentiment"
date: 2015-10-07 00:00
category: reference/moocs
status: draft
parent: ml-foundations
---

* Classification:
    * Spam
    * Tweet Sentiment
* Restaurant review example
    * Need to understand the "aspects" of the review (eg where they saying parts of the restaurant experience was bad or the whole thing?)
    * Look at all reviews and break into sentences.
        * Put sentences into "Sentence Sentiment Classifier".
* Classifier:

    1. Takes in x
    2. Puts it through classified model and returns x's classification.

* Linear classifier
    * Most common type of classifier.
    * Examples:
        * Simple threshold classifier: count words in text and see how many negative and positive words.
    * Problems:
        * Where do we get a list of positive / negative words from?
        * Words have different degrees of sentiment:
            * Great > good
        * Weighing words
        * Single words are not enough.
    * Solution:
        1. Use training data to learn weight of each word.
        2. Add weights to get total score of sentence.

* Decision boundaries
    * In the above example, it'd be the space inbetween positive and negative predictions (0 total weight aka "neutral" sentiment).
    * Is a line: why it's considered a "linear classifier".
    * As you increase number of features, boundary "shape" changes:
        * 2 weights: line
        * 3 weights: plane
        * Many weights: [Hyperplane](../../../../permanent/hyperplane.md)
* Evaluting classification models
    * Similar to regression using training set and test set; check results of training set against test set.
    * Outcomes measured as follows:
        * Error as fraction of mistakes:

                  error = num_of_mistakes / total_num_of_sentences

            * Best possible value is ```0.0```.
        * Accuracy of fractions of mistakes:

                  accuracy = num_of_correct / total_num_of_sentences

            * Best possible value is ```1.0```.
            * Also: ```error = 1 - accuracy``` & ```accuracy = 1 - error```.
* What's a good accuracy?
    * Compare against "baseline" approaches:
        * Random guessing (would have at least a 50% accuracy for two types of labels/classes).
        * "Majority case" (eg: 90% success rate finding spam, but spam constitutes 90% of email): "class imbalance".
    * What does your application need?
* Evaluating classification models
    * Type of mistakes:

              |---------------------------
              | True         |  False    |
                Positive     |  Negative |
              |---------------------------
              | False        |  True     |
                Positive     |  Negative |
              |---------------------------

    * Relationship between these outcomes: [Confusion Matrix](../../../../permanent/confusion-matrix.md).
* How much data does a model need to learn?
    * More data == better, as long as it's good.
    * As you increase data, you reduce test error but never to 0. Gap between model results and 100% accuracy == "bias of model".
    * "Models with less bias tend to need more data to do well".
* Class probabilities
    * Classifier should output a confidence level: ```P(output_label | input)``` (probability of this output label ('positive', 'negative') given this input.
