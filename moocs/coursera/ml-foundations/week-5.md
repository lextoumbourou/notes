# Week 5

* Recommender systems in action

  * "Personalization"
  * Netflix: product recommendations
  * Pandora
  * Facebook friend recommendations
  * Drug-target interactions

* Building recommender with classification

  0. Popularity

    * What are people viewing now?
    * Limitation
      * No personalization

  1. Classification model

    * "Use features of products and users to make recommendations"
    * Features
      * User info
      * Purchase history
      * Product info
      * Other info
    * Classifier takes features as input and return the probability of buying the product
    * Pros:
      * Personalized: considers user info etc
      * Features can capture context: time of day, what user just saw
      * Handles limited user history: age of user etc.
    * Cons:
      * Features may not be available (might not know age etc).
      * Doesn't work as well as "collaborative filtering"

  2. Collaborative filtering

    * "People who bought x also bought y."
    * Matrix C:
      * Store # users who bought both items i & j
        * no items x no items matrix
      * Considered a "symmetric" matrix because ``matrix['comic_book']['bmx'] == matrix['bmx']['comic_book']``
        * "Number of users who bought a comic book and a BMX is the same as users who bought a BMX and a comic book."
    `
    * How it would be used:
      1. User buys a comic book.

      2. Go to comic book row and find other items with the highest row.

* Need to normalize co-occurrence matrix

  * Very popular items "drowns out" other items.
  * One strategy: "Jaccard similarity"

* "Jaccard similarity"

  * Normalize by popularity.
  * Overview:

    1. Count people who purchased i and j
    2. Count people who bought i or j
    3. Divide 1 by 2

  * In code:
  
      ```
      purchased_i.union(purchased_j) / purchased_i.intersection(purchased_j)
      ```
  * Limitations :

    * Doesn't factor in purchase history, only looks at current item.

* (Weighted) average of purchased items

  * Up to here.
