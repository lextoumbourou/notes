# 1. Meet Lucene

## 1.3 Indexing and searching

* Commmon misconception: Lucene is an entire search app, it's the core indexing and searching component.

## 1.3.1 Components for indexing

* Naive approach to search: scan each file for word or phrase.
* Better approach: index text and convert to format you can search quickly.
* Analyze document:
  * Break text down into "tokens"
    * Questions here:
      * How do you handle compound words?
      * Should you apply spell correction?
      * Inject synonyms inlined with original tokens?
      * Stemmer to derive roots from words?
  * Lucene provides heaps of built-in analyzers that give you fine control over process.
  * Can create own analyzer or create arbitrary analyzer chains 

## 1.3.2 Components for searching

* Recall: how well the search finds the relevant docs.
* Precision: how well system filters out irrelevant documents.
* Package called ``QueryParser`` converts user text into query object.
* 3 common models of search:
  * Pure boolean model: docs match or not. No scoring is done.
  * Vector space model: releacne == vector disance between query and doc.
  * Probabilistic model: get probability that doc is a match.
