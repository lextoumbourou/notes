---
title: Mixed membership models for documents
date: 2021-10-30 00:00
modified: 2023-04-08 00:00
status: draft
---

# Mixed membership models for documents

Covers Latent Dirichlet Allocation: assigning documents to multiple topics/clusters at once.

## Motivation

* In previous models, documents are assumed to be assigned to a single topic (though probabilistic clustering model allows for uncertainty).
* Mixed membership models allow for a document to be in 2 topics simultaneously (as a lot of documents generally are).

## An alternative document clustering model

* Builds up to LDA.
* Data representation bag of words instead of TF-IDF.
* Words stored as "multi set": set with dupes.
* First, find prior probability that a doc i is from topic k:

  $$p(z_i = k) = \pi_k $$

  * Basically, how prevalent is each topic across the corpus.
* Then, determine probability of each word within topic and use to compute likelihood for each topic based on word count.

## Components of latent Dirichlet allocation model

* In LDA: every word is assigned to a document.
  * One topic indicator $$z_{iw} \text{ per word in doc i} $$
* Introduces a topic proportion at the document level: how much is this article about science compared to sport (for example).
  * $$\mathbf{\pi_i} = [\pi_{i1}, \pi_{i2}, ..., \pi_{ik}] $$

## Goal of LDA inference

* LDA inputs: words per doc for each doc in corpus.
* LDA outputs:
  * Corpus-wide topic vocab distributions
  * Topic assignments per word.
  * Topic proportions per word.
* Allows you to examine coherence of learned topics:
  * See top words per topics.
  * If they're meaningful, can let you post-facto label topics.

## The need for Bayesian inference via Gibbs sampling

* Introduces a thing call "Gibbs sampling" (though it is not mentioned again in this lecture).
* Revision of K-means:
  * Assign observations to closest cluster centre.
    * $$z_i \text{ <- arg min } ||\mu_j - x_i||_2^2 $$
  * Revise cluster centres.
  * Can be viewed as "maximising a given objective function".
* Revision of EM for MoG
  * E-step: estimate cluster responsibilities.
  * M-step: maximize likelihood over parameters.
* When using bag of words over tf-idf, can still derive EM algorithm: instead of using gaussian likelihood of tf-idf vector, use multinomial likelihood of word counts.
  * $$m_w $$

successes of word w.
  * Called a "mixture of multinomial model".
* LDA model:
  * Could derive EM model, but due to high dimensional space over-fitting tends be a huge problem.
  * Instead: specified as a Bayesian model (when it is, it's usually called "probabilistic latent semantic analysis/indexing").
    * Account of uncertainty in parameters when making predictions.
    * Naturally regularizes parameter estimates in contrast to MLE.
    * Variation on EM used called "variation EM" which introduces an approximation to handle the "intractable expectation" (eh?)

## Gibbs sampling

* Get random hard assignments from a specific distribution iteratively.
* Intuitive and easy to implement.
* Joint model probability: probability of observations given variables / parameters and probability of variables / parameters themselves
  * Example: look at probability of "computer" within technology topic and probability of "computer" in entire corpus.
  * Not considered an optimisation algorithm: jmp will jump around bit as iterations increase but should eventually provide "correct" Bayesian estimates.
* Predictions:
  1. Make prediction for each randomly assigned variables/parameters (full iteration).
  2. Average predictions for final result.

* Parameter or assignment estimate:
  * Look at snapshot of randomly assigned variable/parameters that "maximises joint model probability".

## A standard implementation of Gibbs sampling

* Step 1: Randomly reassign all words in a document to some topic based on: doc topic proportions and topic vocab distributions
  * firstly create a responsibility vector for each word as follows:
    * ```word_probability['EEP', 'topic_2'] = prior_probability('topic_2') * probability('EEP', given='topic_2') / sum(prior_probability(topic) * probability('EEP', given=topic) for topic in topics)```
    * $$r_{iw2} = \pi_{i2} * P(\text{"EEG"} \mid z_{iw} = 2) $$
  * then picked out a topic at random for the word based on the probabilities just defined.
* Step 2: Randomly reassign doc topic proportions based on assignments defined in the last step.
* Step 3: Repeat for every doc.
* Step 4: Randomly reassign topic vocab based on distributions based on assignments in step 1 for entire corpus.

## Collapsed Gibbs sampling in LDA

* Based on special structure of LDA model, can sample just indicator variable $$z_iw $$
  * No need to sample corpus-wide topic vocab distributions
  * Per-doc topic proportions
* Can lead to much better performance.
* Randomly reassign $$z_iw $$based on current assignments $$z_jv $$

of all other words in document and corpus.

## A worked example for LDA: initial setup

1. Start with a document: "Install Windows on her laptop".
2. Assign each word to a topic randomly: {Install: 1, Windows: 2, her: 3, laptop: 2} (can initialise a lot smarter).
3. Form local counts:

    ```
           | Topic 1 | Topic 2 | Topic 3
    Doc i  |       1 |       2 |       1
    ```

4. Build corpus wide statistics.

    ```
            | Topic 1 | Topic 2 | Topic 3
    Install |      20 |       0 |       4
    Windows |      1  |      12 |       10
    ```

5. Iterate through every word in the entire corpus and resample.
    * When resampling a word, need to remove it from the local counts and corpus wide tables first.
     * Then, resample based on $$p(z_iw \mid \text{every other } z_jw \text{ in corpus, words in the corpus}) $$

## A worked example for LDA: deriving the resampling distribution

* Figure out how much doc likes topic:
  * $$n_ik $$

= current assignments to topic k in doc i.

  * $$N_ik $$

= words in doc i.

  * $$\frac{n_ik + \alpha}{N_i - 1 + K\alpha} $$
* Figure out how much topic likes word:
  * $$\frac{m_{word,k} + \alpha}{\sum_{w \in V} m_{w,k} + V\alpha} $$
* Then multiply the 2 probabilities "How much doc likes topic" * "how much topic likes word"
  * Then pick topic randomly using this probability
* Update the 2 tables as per last section.

## Using the output of collapsed Gibbs sampling

* Not really sure what this lecture is saying?
* Look at best sample of {z_iw} can infer:
  * Topics from conditional distribution
  * Document "embedding": form topic proportion vector for document.
