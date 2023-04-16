---
title: Week 1 - Introducing Recommender Systems
date: 2017-03-04 00:00
category: reference/moocs
status: draft
parent: recommender-systems-introduction
---

## Intro to Course and Specialization

* Broken into 4 courses:
    * Non-personalized and content-based.
    * Nearest-neighbor collaborativ filtering.
    * Evaluation and metrics.
    * Matrix factorization and advanced techniques.
* Capstone project:
    * Case study analysis - design best recommender for a business use case.

## Predictions and Recommenders

* Recommendations have less authority than predictions: "maybe you'll like this", not "I'm sure you'll like this".
* Predictions:
    * Pro: help quantify item.
    * Con: could provide something falsifiable.
* Recommendations:
    * Prod: could provide good choice as default.
    * Con: if peeps think it's "top-n", could stop them exploring.
* Recommenders could be considered "softer sell" eg book stores have some books facing forward, others just spine. TVs in more prominent positions.

## Taxonomy of Recommenders

* Domain of recommendation: "what's being recommended?"
    * News articles
    * Products
    * Matchmaking
    * Sequences (musical playlists)
* Interesting property:
    * Is it new items (movies, books)?
    * Re-recommend old ones (groceries, music)?
* Examples of recommenders:
    * Google search results.
* Purposes of recommdations
    * Sales.
    * Education of user/customer.
        * Tip of the day in Office products.
    * Building community around products or content.
* Recommendation context
    * What's the user doing when making recommendation?
        * Shopping
        * Listening to music
    * How does the context constrain the recommender?
        * Groups, automatic consumption, level of attention?
* Whose opinion?
    * Recommenders are usually based on somebodies opinion:
        * Experts, other users, etc.
    * "Phoaks" (People helping one another know stuff)
* Personalization level
    * Generic / non-personalized
        * Same recs for all.
    * Demographic
        * Women get different recs then men etc.
    * Ephemeral
        * Match what you're currently doing.
    * Persistent
        * Interests over time.
* Privacy and trustworthiness
    * Who knows what about me?
        * Personal info reveal.
        * Identity.
        * Deniability of pref
    * Is the rec honest?
        * Biases built-in by operator ("business rules")
        * Vulnerability to external manupulation
            * Example: higher scores for new movies. Are movie studies "hacking" the results?
        * Transparency of "recommenders": reputation
* Interfaces
    * Types of output
        * Predictions
        * Recommendations
        * Filtering
        * Organic vs explicit presentation
    * Types of input
        * Explicit
            * Being asked to review things.
        * Implicity
            * How often have you looked at a certain page?
* Recommendation algorithms
    * Non-personalized summary stats
    * Content-based filtering
        * Info filtering
        * Knowledge-based
    * Collaborative filtering
        * User-user
        * Item-item
        * Dimensionality reduction
    * Other
        * Critique / interview based recs.
        * Hybrid techniques.

## Taxonomy of Recommenders 2

* Notions every recommender needs:
  * Users
      * Users may have attributes (demographics).
  * Items
  * Ratings
      * Users make rating on items somehow (implicity and explicit).
* Non-personalized summary stats:
    * External community data:
        * Best selling, most popular, trending stuff.
    * Summary of community ratings:
        * Best liked.
        * Pull out ratings for some item and take average.
    * Examples:
        * Zagat restaurant ratings
        * Billbard music rankings.
        * TripAdvisor hotel ratings.
* Content-based filtering:
    * User ratings x item attributes => model
    * Model applied to new items via attributes
        * User liked articles about soccer, future articles about soccer may be recommender.
        * Fan of certain genres of movies.
        * Fan of movies with certain actors.
    * Alternative: knowledge-based
        * Item attributes form model of item space.
            * Users navigate/browse that space.
    * Examples:
        * Personalized news feeds.
        * Artist or genre music feeds.
* Personalized Collaborative Filtering
    * Use opinions of others to predict/recommend.
    * User model - set of ratings
    * Item model - set of ratings
    * Common core: sparse matrix of ratings
        * Fill in missing values (predict)
        * Select promising cells (recommend)
    * Several different techniques.
* Collaborative Filtering Techniques
    * User-user
        * Get "neighbourhood" of people with similar tasts.
            * Could only select "trustworthy" people.
    * Item-item
        * Compute similarity amongst items using ratings.
        * Use ratings to triangulate for recs.
    * Dimensionality reduction
        * Intuition: taste yields a lower-dim matrix.
        * Compress and use taste representation.
* Note on evalution:
    * Will spend time on evaulation:
        * Accuracy of predictions.
        * Usefulnes of recommendations: correctness, non-obviousness, diversity.
    * Computational performance.

## Tour of Amazon

* Dimesions of analysis:
    * Recommendations based on implicit purchase data.
    * Personalization level: one product at a time.

## Recommender Systems: Past, Present and Future

* Before recommender systems:
    * Manual personalization
    * Cross-sales and early product associations
    * Product search.
* Tech bubble:
    * During: recommenders were seen as "key technology"
    * After: recommenders were put in context of the things the business was actually trying to do.
* Wave 2: The Netflix Prize
    * Netflix $1M prize
    * Recommendation as app area for data mining, machine learning
    * Rapid growth in field
    * New techniques:
        * Algorithm stacking
        * New matrix factorization techniques
* Mature realizations:
    * Predictions and top-n algos are limited
        * Limitations to how good recommendation engines are with people driving them.
* State of field today:
    * Lots of well-known algos
    * Effective recs still a "craft"
        * Exploring data.
        * Understanding usage cases and value proposition
    * Still largely focuses on business apps
        * Creativity
        * Dream of consumer-owner not realized.
* Looking forward:
    * Hard problems unsolved:
        * Temporal recommendations: "what should you consume next?"
        * Recs for education.
        * Low-frequency, high-stakes recs: can we help you find a house or other things you don't have rating for?
    * Recognized speciality that brings ML, business + marketing, human-computer interaction etc.
* Promising directions:
    * Context.
    * Sequences: music, education etc.
    * Lifetime value.

## Introducing the Honors Track

* Involves implementing algorithms using LensKit toolkit.
* Software required:
    * Java dev kit.
    * Dev environment (Eclipse, IntelliJ etc)..
    * Data analysis software: Excel, R PyData etc.
* LensKit handles external stuff like I/O, setting up data for evaluation etc.
* Written in Java because a) lots of people know it and b) can achieve good performance.
