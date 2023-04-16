---
title: "Week 5 - Recommending Products"
date: 2015-10-07 00:00
category: reference/moocs
status: draft
parent: ml-foundations
---

* Recommender systems in action
    * "Personalization"
    * Netflix: product recommendations
    * Pandora
    * Facebook friend recommendations
    * Drug-target interactions
* Building recommender with classification
    * Popularity
        * What are people viewing now?
        * Pros:
            * Rudimentary solution to the "cold-start problem": what to show the user when you know nothing about them.
        * Limitations:
            * No personalization
    * Classification model (like sentiment analysis)
        * Considered supervised learning (?).
        * Potential features:
            * User info.
            * Purchase history.
            * Product info.
            * Other info.
        * Classifier takes features as input and return the probability of buying the product
        * Pros:
            * Personalized: considers user info etc
            * Features can capture context: time of day, what user just saw
            * Handles limited user history: age of user etc.
        * Cons:
            * Quality of information may be low.
            * Features may not be available (might not know age etc).
            * Doesn't work as well as "collaborative filtering"
* Collaborative filtering
    * "People who bought x also bought y."
    * Requires co-occurrence matrix:
        * Store # users who bought both items i & j
        * In code:

                  purchased_items_pos                  =  ['Nike Air', 'Reebook Pump', 'Mercedes Pumas']
                  purchases_also_made['Nike Air']      =  [53,         2,               3] 
                  purchases_also_made['Reebook Pump']  =  [2,          22,              6]
                  purchases_also_made['Mercedes Pumas] =  [3,          6,               100] 

        * Considered a "symmetric" matrix because

                    matrix['Nike Air']['Mercedes Pumas'] == matrix['Mercedes Pumas']['Nike Air']

            * "Number of users who bought Nike Airs and Mercedes Pumas is the same as users who bought a Mercedes Pumas and Nike Airs."
    * How it would be used:

        1. User buys Nike Airs.
        2. Sort the Nike Airs row by "also made" purchases and display the top few.

                recommended = sorted(purchases_also_made['Nike Air'])[:recommendation_count]

    * Need to normalize co-occurrence matrix
        * Very popular items "drowns out" other items.
        * One strategy: "Jaccard similarity"
    * "Jaccard similarity"

        ![Normalise by Popularity](/_media/ml-foundations-normalize-by-popularity.png)

      * Normalize by popularity.
      * Overview:

          1. Count people who purchased i and j
          2. Count people who bought i or j
          3. Divide 1 by 2

      * In code:

                purchased_i.union(purchased_j) / purchased_i.intersection(purchased_j)

    * Limitations:
        * Doesn't factor in purchase history, only looks at current item.
            * Solution: "weighed average of purchased items".
  * Weighted average of purchased items
    * For all potential recommendable products, go through and calculate similarity score based on each of user's purchase history (maybe weigh recent purchases higher).
    * In code:

            user_inventory = {'book', 'hat', 'shoe'}
            for product in potential_recommendations_in_inventory:
              score[product] = average([some_similarity_score(p) for p in user_inventory])

            highest_j = sort(score)

    * Limitations:
      * Doesn't use a variety of available features:
        * Context
        * User features
        * Product features
      * "Cold start problem" - dealing with user's with no purchase history.

  3. Discovering hidden structure by matrix factorization

    * Matrix of movies x users:

              movie_pos     =  ['Taxi Driver', 'Goodfellas', 'Casino']
              rating['John'] = [None,          8,            6      ]
              rating['Bill'] = [3,             3,            5      ]
              rating['Bob']  = [7,             2,            None   ]

    * Attempt to fill in unrated movies with user's ratings and rating of other users.
      * Point: guess the rating a user would give to movies they haven't seen.
    * Overview:

      1. Describe movie ``v`` with topics ``R(v)``: how much action, romance, drama:

               movie_genres                       = ['action', 'romance', 'drama']
               movie_genres_amount['Taxi Driver'] = [0.3,      0.1,        0.5]  # R(v)
               movie_genres_amount['Goodfellas']  = [0.2,      0.1,        0.7]  # R(v)

      2. Describe how much user ``u`` with topics ``L(u)``.

               user_genres_prefered['John'] = [2.5,       0,         0.9]  # L(u)
               user_genres_prefered['Bill'] = [5,         2,         10]   # L(u)
               
      3. ``Rating(u, v)`` is the product of two vectors:

                ratings = lambda u, v: sum([x * y for x, y in zip(user_genres_prefered, movie_genres_amount)])

      4. Sort movies by ```Rating(u, v)```

* Predictions in matrix form
    * Looking at score for ```Rating(u, v)``` == ```<Lu, Rv>``` < element-wise product and sum.
    * Can get the ```u```th from the ```L``` matrix row and multiple by the ```v```th row from the ``R`` matrix column.

        ![Predictions in matrix form](/_media/ml-foundations-predictions-in-matrix-form.png)

* Discovering hidden structure by matrix factorization
    * Using only observed rating (black squares), we want to estimate L and R matrices.
    * Look at predictions compared to actual observed rating (similar to Regression):

              RSS(L, R) = (Rating(u, v) - <Lu, Rv>) ** 2 + sum([(u_prime, v_prime) for pairs if Rating(u_prime, v_prime)

    * Reason called a "Matrix Factorization Model" because taking matrix and approximating with factorization.
    * Many efficient algorithms for factorization.
    * Limitations:
        * Cold start problem: new user or new movies (no ratings).
* Featurized matrix factorization
    * Features capture context
        * Time of day
        * What I just saw
        * User info
        * Past purchases
    * Discovered topics from matrix factorization capture "groups of users" who behave similarly.
         * Women from Seattle who teach and have a baby.
    * Combine to mitigate cold-start problem
        * Ratings for a new user from features only.
        * As more information about user is discovered, matrix factorization *topics* become more relevant.
    * "Blending models"
        * Winning team of Netflix prize blended over 100 models.
* Performance metric for recommender systems
      * Classification accurary

          * Problems:
              * Major class: recommend no items.
              * Can only get a limited subset of correctly classified items, since user has limited attention.
    
        * Solution:
            * Metrics called "Precision" and "Recall".
    
      * Recall
        * Find all liked and figure out how many were recommended.
        * Formula: ```liked_and_shown / liked```

                  items_recommended = set(['Shoes', 'Jeans', 'Hat', 'Pants'])
                  items_liked       = set(['Jeans', 'Hat', 'Glasses'])
                  recall = len(items_recommended.intersection(items_liked)) / len(items_liked)  # 2 / 3
    
      * Precision
    
          * Get all recommended items and figure out how many were liked.
          * Formula: ```liked_and_shown / shown```
          
                  precision = len(items_recommended.intersection(items_liked)) / len(items_recommended)  # 2 / 4

  * Optimal recommenders
      * "How do you maximise recall?" - just recommend everything.
          * Small precision.
      * What is optimal recommender:
          * Recall = 1 (everything liked was recommended).
          * Precision = 1 (everything recommended was liked).
* Precision-recall curve
    * Input: a specific recommender system
    * Output: Algorithm-specific precision-recall curve
    * Optimal recommender precision / recall curve:

        ![Perfect Precision Recall Curve](/_media/ml-foundations-perfect-precision-recall-curve.png)

    * Realistic recommender:

        ![Realistic Precision Recall Curve](/_media/ml-foundations-realistic-precision-recall-curve.png)

    * Comparing algorithms:
      * Largest "area under the curve" (AUC).
      * Set desired recall and maximise precision (precision at k).
* Recommender systems ML block diagram
    * Training data: user, products, ratings table
    * Feature extraction:
        * (user_id, product_id).
        * Also may include other features like gender, age, product description etc.
    * Goal (y-hat): predicted rating for user, product pair.
    * Model: matrix factorization.
        * Set of features for every user (how much user likes action, comedy etc).
        * Set of features for every product (how much a movie action, comedy etc).
    * Quality metric:
        * RSS (compare error between predicted ratings and observed values).
