# Week 3 - Content-Based Filtering Using TFIDF

## Introduction to Content-Based Recommenders

* Basic idea: there are "stable" preferences people that that can be measured by content attributes:
  * News - politics, sports, fashion etc.
  * Clothing - bright, dull etc.
  * Movies  - horror, drama etc.

* Core ideas:
  * Use attributes to model items.
  * Figure out user prefs with said attributes.
  * Recommend stuff based on user prefs finding other items with attributes.

* Building a vector of attributes or keyword preferences.
  1. User could build their own profile.
    * Maybe a bit awkward to make them create the whole profile, but allowing to edit can work.

  2. Can build profile from actions.
    * Clicks, buys, bookmarks etc.

  3. Infer user profile from explicit user ratings.
    * Map item preference to attribute preference.

  4. Merge actions/explicit into infer from rating (explicit and implicit).

* How to use preferences:
  * Given a vector of keyword preferences:
    * Can we just add likes and dislikes?
    * Can we figure out which keywords are more and less relevant?
      * TF/IDF.

* Case-based recommendation
  * Build DB of "cases" to attributes (camera price, zoom, pixels).
  * Query based on atttributes to get a relevant case.
  * Open issue: many ways to structure interactions.

* Ask Ada example:
  * Interview process to determine purchase preferences.

* Knowledge-based recommender.
  * Case-based example with navigation interface.
  * FindMe systems (eg Entree).

* Content-based techniques challenges and drawbacks:
  * Needs well-structure attributes that align with preference.
    * Painting example: hard to reduce paintings to list or attributes, don't want more blue more oil etc.
      * Can work with community tagging and analysis.
  * Need reasonable distribution of attributes across items.
    * Doesn't work if all items are the same re attributes.
  * Doesn't find surprising connections
    * Lemon goes well with chocolate but doesn't work if you just say "what's like chocolate".
  * Harder to find complements than substitutes.

* Take-away lessons
  * Many ways to recommend based on content (product attributes):
    * Over time, can build up a profile of user's content preferences.
    * Shorter term: can build database of "cases" and navigate through them.
  * Content-based techniques don't need as large a set of users (just good item data).
  * Cbt is also:
    * Good at finding substitutes.
    * Good at helping navigate for a purchase; good explainability.

## TFIDF and More

* Primitive search engine might look at a search for "The Great War" and do a document search for The or Great or War.
  * Might return document that says "the" a lot, over one that has "The Great War" once.
* Enter TFIDF:
  * Term Frequency * Inverse Document Frequency:
    * Number of occurences of a term in the doc.
    * Usually simple count.
  * Inverse Doc Frequency:
    * How few docs contain this term.
    * Typically: log (#docs / #docs with term).
      * Logarithm is the most effective to get a large number of documents into a useful scale.
  * Automatic demotion of common terms.
  * Promotes core over incidental.
  * Fails when core terms/concepts are used much in document (legal contracts).
  * Poor searches (synomyns etc).
* How does TFIDF apply to content-based filtering?
  * Can be used to profile doc/object.
    * Movie can be weighted vector of tags.
  * TFIDF profiles can be combined with rating the create future user profiles and match against docs.

* Variants and alternates to TFIDF:
  * 0/1 bool frequency (occurs above threshold).
  * Logarithmic frequency (log (tf + 1)).
  * Normalized frequency (divide by doc length).
    * Words occuring more in longer documents need to be considered.
  * BM25 (aka Okapi BM25) is ranking function used by search engines:
    * Include frequency in query, document, num documents.
    * Variants with different weghts: BM11, BM25.

* Other considerations:
  * Phrases + n-grams:
    * "Computer science" != "computer" and "science".
    * Adjacency: phrase queries.
  * Significance in documents:
    * Titles, headings vs main document.
  * General document authority:
    * Pagerank and similar approaches.
  * Implied content:
    * what if document never actually mentions the thing you're searching for?
    * usage of links, usage, synomyns etc.

## Content-based Filtering: Deeper Dive
