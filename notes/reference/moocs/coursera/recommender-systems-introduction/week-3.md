---
title: Week 3 - Content-Based Filtering -- Part I
date: 2017-03-04 00:00
category: reference/moocs
status: draft
parent: recommender-systems-introduction
---

## Introduction to Content-Based Recommenders

* Basic idea: there are "stable" preferences people that that can be measured by content attributes:
    * News - politics, sports, fashion etc.
    * Clothing - bright, dull etc.
    * Movies - horror, drama etc.
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
    * Query based on attributes to get a relevant case.
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
        * Number of occurrences of a term in the doc.
        * Usually simple count.
    * Inverse Doc Frequency:
        * How few docs contain this term.
        * Typically: log (`#docs / #docs with term`).
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
        * Words occurring more in longer documents need to be considered.
    * BM25 (aka Okapi BM25) is ranking function used by search engines:
        * Include frequency in query, document, num documents.
        * Variants with different weights: BM11, BM25.
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

* Keyword vector
    * Make each keyword a dimension of the vector.
    * Each item has a vector.
    * Each user has a vector.
    * Recommendations take items that are closest to user vectors.
    * Can reduce size of keyword space using stemming and stop words etc.
    * Vector space model: https://en.wikipedia.org/wiki/Vector_space_model
* Represent an item through a k/w vector:
    * Could be simple 0/1 (keyword applies or not).
    * Simple occurence count.
    * Use TFIDF?
    * Consider document length?
    * Usually want to normalise vector, so it's human understandable.
* Consider the movie/tag case
    * Do w consider tags yes or no?
        * Actor tags only really need to be tagged once: it either has Arnold S or it doesn't.
        * Other tags may use frequency for description.
    * Do we care about IDF?
        * Less common actors more significant than stars?
        * "Prison scene" may be more telling to preference than more common tags like "car chase" or "romance".
* Formalisation:
    * Each tag $t \in T$ can be applied to an item or not.
    * $T_i$ is the set of tags applied to an item $i$ or $t_i$ is a 1/0 vector of tags.
    * Have a case where user apply tags to items (each user could apply each tag 0 or 1 times to an item). $t_{ui}$ is a tag application of a tag by a user to an item.
    * Have a case where tags are multiply applied (by users or algorithms) and $\vec{t_i}$ is a weighted vector of tags.
* From items to user profiles (1)
    * Vector space thinks liking is the same as importance:
        * Good for some use cases like movies but not for other use cases.
* How do we "accumulate" profiles?
    * Add together the item vectors?
        * Normalise first?
        * Weigh vectors somehow?
* Factoring in ratings:
    * Unary: add item vectors of items we rate.
    * Binary: add positives, subtract negatives.
    * Unary with threshold: just put items above a certain rating (aka above 7) in preferences.
    * Weigh only positives: give higher weight to positive things.
    * Weigh including negatives: negative weight for low ratings.
* How do we update profiles (new ratings)?
    * Don't: recompute each time (could be wasteful).
    * Weight new/old similarly.
        * Special case for changed rating; subtract old.
    * Decay old profile and mix in new (maybe have profile dominated by most recent movies).
* Computing predictions:
    * Predictions is cosine of angle between 2 vectors: profile and item.
        * Cosine starts at its maximum, +1, when 2 things have same angle and grows negatively as angle grows.
    * Computed as dot product of normalised vectors.
    * Cosine ranges between -1 and 1 (0 and 1 if all positive values in vectors) - closer to 1 is better.
    * Top-n or scale for rating-scale predictions.
* Strengths of this approach
    * Entirely content-based
    * Understandable profile
    * Easy computation
    * Flexible: can integrate with queries and case-based approaches.
* Challenges:
    * Figure out the right weights and factors
        * Is more *more* or just reiteration of same.
        * Deal with ratings.
* Limitations:
    * Simplified model, doesn't handle interdependencies:
        * I like horror movies about supernatural but not supernatural documentaries.
        * "Breaking things down into attributes doesn't always work".
* Summary
    * Can do content-based filtering based on assessing the content profile of each item.
    * Build user profiles by aggregating the item profiles of things they've bought or rated or consumed etc.
    * Then, you can evaluate unrated items using the user profiles by taking the vector cosign.

## Advanced Content-Based Techniques and Interfaces

### Entree Style Recommenders - Robin Burke Interview

* Problem Entree Style Recommenders were solving:
    1. Look at navigating in spaces with lots of choices of restaurants.
    2. Coming from background of "case-based reasoning".
        * Branch of artifical intelligence that deals with large chunks of data: entire plan instead of little steps.
    3. Restaurants appeared "case-based" to Entree.
* How the the recommender works:
    * Find the similarity between objects you have and objects that are out there in the DB.
        * Give them restaurant you have and will return other restaurants like it in Chicago.
    * Properties arranged hierarchiely:
        * Cusines first.
        * Price.
        * Quality of experience.
    * Restaurant similaries are "asymetric".
        * Restaurant A may be similar to restautant B because it's cheaper but not vice versa.
            * People don't want to be recommended a more expensive restaurant.
    * Built a symantic network for cuisines: for any 2 cuisines, find the different between them.
    * Data scraped off the web.
        * Data got old pretty fast: restaurants come and go.
    * Investigated multiple user models:
        1. Keep track of user critiques and only show stuff that you think they'd like.
            * With enough constraits became near impossible to show anything: don't know which constraint to relax.
            * People found it frustrating.
            * System should always show you something.
        2. User model to do with similarity metric itself.
            * If you have lots of money, you'd put price low in hierarchy model.
            * Need to detect if someone is rich etc.
        3. When with default similiarty metric.
    * Core idea: let user make up pretend restaurant then put attributes into the database to see similar restaurants but in new cuisine type.

### Case-based reasoning - Interview with Barry Smyth

* Case-based reasoning is a form of "memory-based reasoning"
    * Try to use past problems and solutions to solve new problems.
* When faced with solving a new problem, tries to find a similar case and reuse its solution.
* Useful in domains without strong problem solving domains.
* Case-based reasoning is similar to content-based recommendations: rely on characteristics of the items.
    * Difference is that case-based reasoning is that items are described with well structure descriptions.
* Case-based reasoning represents items as structured, feature-based representations.
* Example of item feature values:
    * Restaurant might characterised by cuisine type, price, location etc.
    * Very well defined set of structural features.
* Tension between relevance and diversity:
    * Do you want diverse recommendations or relevant?
    * How can you cause retrieval of more diverse cases while still being relevant?
* Hot topics in case-based recommendation:
    * "Opinion mining"
        * Look at reviews and see the sentiment towards particular features eg MacBook Air battery life etc.

### Dialog-based Recommenders - Interview with Pearl Pu

* Users provide feedback, called "critiques" on recommended items.
* Rating based vs critiquing based:
    * Rating based: user preferences are represented as score: one to five.
    * Critque based: allow users to specify preferences in more depth.
* Useful in domains where available set of choices change a lot, eg laptops change so much, preference cannot be captured with simple ratings.
* Weight preferences based on importance.
* Uses a method called "multi attibute utility theory" or MAT.
    * Some preferences have common sense utility direction: lighter is better, cheaper is better.
    * Others are very personal like brands.
* Strength of MAT is it's easy to compute, weakness is that it assumes independance between features.
