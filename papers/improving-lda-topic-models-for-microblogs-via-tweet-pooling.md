# Improving LDA Topic Models for Microblogs via Tweet Pooling and Automatic Labeling

## Abstract

* LDA good for news articles and academic papers but not great for microblog content.
* Paper improves topic modelling for microblogs without modifying LDA.
* Uses Tweet pooling by hashtags to achieve big improvements, contrasted with unmodified LDA.

## 1. Introduction

* Topic models help uncover patterns of textual data.
* Mixed-membership assumption: documents belong to multiple topic groups.
  * Represented as probabilistic distributions over words.
  * LDA is an example of probabilistic topic model.

* Topics learned from LDA are a multinomial distribution over words.
  * Top-10 words are used to identify found topics.
  * Twitter gives mostly incoherent topics.

* Twitter poses 3 problems to LDA:
  1. Short posts.
  2. Mixed contextual clues like URLs, tags and Twitter names.
  3. Informal language, misspelling etc.

* Proposable to fix language problem: [Automatically Constructing a Normalisation Dictionary for Microblogs](http://www.aclweb.org/anthology/D12-1039).
  * Also need complimentary approach to deal with content size.
* Paper compares performance of methods across 3 datasets.
  * Use various evaluation metrics, ability of learned LDA topics to reconstruct known clusters.
  * Interpreability of topics vai statistical information measures.
* Pooling tweets by hashtag yields superior performance for all metric on all datasets.

## 2. Tweet Pooling for Topic Models

* Goal of paper: obtain better LDA topics without touching LDA too much.
* Address challenge by aggregating tweets into "macro-documents" for use as training data.
* Tweet pooling schemes available:
  1. Basic scheme - Unpooled: treats each Tweet as a single document and trains LDA on all Tweets. Baseline for comparison to pooled schemes.
  2. Author-wise Pooling: pooling Tweets according to author. Standard way to aggregate Twitter data; superior to unpooled Tweets.
  3. Burst-score wise Pooling: find "bursts" of term frequencies aka trending topics and aggregate over them.
      $$ \text{burst-score}(R, d) = \frac{|M(R, d) - Mean(R)|}{SD(R)} $$ 
  4. Temporal pooling: pool all tweets within the same hour of some major event.
  5. Hashtag-based pooling: pool based on hashtags. If contains multiple hashtags, the document gets added to multiple pools.

## 3. Twitter dataset construction

* Description of 3 datasets:
  1. Generic dataset: 359,478 tweets from 2009-01-11 to 2009-01-30. General dataset with general terms.
  2. Specific dataset: 214,580 tweets from 2009-01-11 to 2009-01-30. Tweets that refer to named entities.
  3. Event dataset: 207,128 tweets from 2009-06-01 to 2009-06-30. Tweets pertaining to specific events.

## 4. Evaluation metrics

* Purity
* Normalized Mutual Information (NMI)
* 

## 5. Results for pooling schemes

* Purity score:
  * Generic: Hashtag and author perform equal best.
  * Specific and events: hashtag pooling performs best.
* NMI score:
  * Generic: hashtag and author perform equal best.   
  * Specific and events: hashtag pooling performs best.
* PMI score:
  * Hashtag performs best on all. 

