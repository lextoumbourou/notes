# Week 1

## Intro to Course and Specialization

* Broken into 4 courses:
  * Non-personalized and content-based.
  * Nearest-neighbor collaborativ  filtering.
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
