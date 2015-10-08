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

    * Limitations:

      * Doesn't factor in purchase history, only looks at current item.

  * (Weighted) average of purchased items

    * For all potential recommendable products, go through and calculate similarity score based on each of user's purchase history (maybe weigh recent purchases higher).

    * In code:

        ```
        user_inventory = {'book', 'hat', 'shoe'}
        for product in potential_recommendations_in_inventory:
          score[product] = average([some_similarity_score(p) for p in user_inventory])

        highest_j = sort(score)
        ```

    * Limitations:

      * Doesn't use a variety of available features:

        * Context
        * User features
        * Product features

      * "Cold start problem" - dealing with user's with no purchase history.

  3. Discovering hidden structure by matrix factorization

    * Matrix of movies x users.
    * Attempt to fill in unrated movies with user's ratings and rating of other users.
      * Point: guess the rating a user would give to movies they haven't seen.
    * Overview:

      1.  Descrive movie ```v``` with topics ```R(v)```: how much action, romance, drama:
           ```
           movie_genres = ['action', 'romance', 'drama']
           taxi_driver_topic = [0.3, 0.001, 1.5]  # R(v)
           ```

      2. Describe how much user ``u`` with topics ``L(u)``.

           ```
           user_genres = [2.5, 0, 0.9]  # L(u)
           ```
               
      3. ``Rating(u, v)`` is the product of two vectors:

            ```
            Rating = lambda u, v: sum([x * y for x, y in zip(u, v)])
            ```
      4. Sort movies by ```Rating(u, v)```

* Predictions in matrix form
  
  * Looking at score for ```Rating(u, v)``` == ```<Lu, Rv>``` < element-wise product and sum.
  * Can get the ```u```th from the ```L``` matrix row and multiple by the ```v```th row from the ``R`` matrix column.
    
  <img src="./images/predictions-in-matrix-form.png"></img>

* Discovering hidden structure by matrix factorization 

  * Up to here.
