---
title: Week 2 - Non-Personalized and Stereotype-Based Recommenders
date: 2017-03-04 00:00
category: reference/moocs
status: draft
parent: recommender-systems-introduction
---

## Non-Personalized and Stereotype-based Recommenders

* Why non-personalised?
    * Handling new users (cold start problem).
    * Simple but effective.
    * Online communities with common interests (Hacker News, Reddit etc).
    * Some apps cannot be personalised.
* Long history of recommendation in print.
    * Editorial opinions
    * Surveys and aggregate reviews (Zagat)
* Aggregate implicit data:
    * Billboard top 200 charts: how many times was album purchased.
    * "Popular now" on news sites
* Weak personalisation: you know only a bit about the user:
    * Location
    * Age, gender, nationality, ethnicity.
* Covered in the module:
    * Summary stats.
    * Lightly personalised recommendations.
    * Finding related items.

## Summary Statistics I

* The Zagat guide:
    * Started out as small booklet of compiled reviews from restaurants in NY.
* If computing score for Zagat, what should you use?
    * Popularity? How many people are recommending it.
        * Lots of people eat at KFC: would that be a top rating?
    * Average rating?
        * Few ratings could skew the results.
    * Probability of you liking it (what does that even mean)?
* Zagat formula:

      Rating = {0, 1, 2, 3}
      Score = Round(Mean(ratings) * 10)
      
    * Don't include things like McDonalds where popularity would be a dominate factor.
    * In 2016, they switched from 0-3 and 0-30 to 5-stars (with halfs) and averaging to a tenth of a star.
* Conde Nast Traveler tallies % of people who rate a hotel, cruise as "very good" or "excellent"
    * Means you could potential see results where lots of people really loved it and some didn't, as oppose most people thought it was pretty okay.
        * Averaging scores doesn't give you this level of granularity.
* Amazon Customer Reviews
    * Average for star rating, but also shows how many people voted for each number of stars (90% 5 stars, 5% 4 starts, 5% 1 star etc).
* Breaking it down:
    * Popularity is an important metric.
    * Averages can be misleading:
         * Consider summin % who liked.
         * Normalising user ratings: if all reviews are 1, 2 and 3s and someone else 4s and 5s, maybe the 3s for the first are the other's 5s.
         * Credibility of individual ratings:
             * "Imposter reviews"
    * More data is better (but don't you overwhelm people).
        * Average, count, distribution etc.
* What's missing?
    * Who you are:
        * Looking for new songs, might not look for songs popular amongst 15-year old girls
    * Your context:
        * Ordering ice-cream and want most popular sauce, do you want to be recommended tomato sauce?
* Zagat fans say guide is getting worse: Mediocre restaurants have good scores, good restaurants have low scores.
    * Self-selection bias:
        * If you gave it a negative review one year, you probably aren't going to review it next year. Score goes up.
    * Increased diversity of raters:
        * Different reasons for changing ratings.
* Take-away lessons:
    * Non personalized popularity stats or averages can be effective in right application.
    * Good to show count, average and distribution together (ala Amazon).
    * Ranking: could average % who score above threshold.
    * Personalisation can address limitiations.

## Summary Statistics II

* Objectives:
    * Understand how to compute and show predictions.
    * Learn how rank items with sparse and time-shifting data.
    * Understand several points in design space for prediction and recommendation and some of their tradeoffs.
* Example - Reddit:
    * Provide non-personalised news articles based on stories peeps lke.
* How to compute aggregate preferences?
    * Average rating / upvote proportion:
        * How many people upvoted it (I'm guessing the time sensitiveness is an issue here).
        * Doesn't show popularity
    * Net upvotes / # of likes (upvotes - downvotes).
        * No controversy.
    * % of people who have rated above some threshold (ala Conde Nast).
    * Full distribution.
* Goal of display:
    * To help users decide to buy/read/view the item.
* Ranking:
    * What do you put at the top of your search results?
    * Don't have to rank by predictions:
        * Too little data (only one 5-star rating).
        * Score may be multivariate (histogram).
        * Domain or business considerations:
            * Item is old (news site for example).
            * Item is unfavored (business reasons for example).
* Ranking considerations:
    * How confident are we that this item is good?
    * Risk tolerance:
        * High-risk, high-reward
            * Maybe someone will find something unexpected and good.
        * Conservative recommendation
            * Can be pretty sure that's what the user wants - could be boring?
    * Business considerations: what exactly is your site trying to recommend.
* Damped means
    * Useful for solving problems of few ratings.
    * Assume everything is average: ratings are evidence of non-averageness.
    * In Python:

      ``(sum(all_ratings_available_for_items) + k * average_ratings_for_all_items) / (num_ratings + k)``

    * k controls strength of evidence required: the higher k, the more ratings required before the damping stops taking effect.
* Confidence intervals:
    * (disclaimer: I *think* this is what is being said here) Based on the data you've got, with some margin of error, the rating would be between 3.5 and 4.2.
    * Choice of bound affects risk/confidence.
        * Taking the lower is conservative, but you're pretty sure it's right.
        * Upper is mor risky, but could result in some great results.
    * Reddit uses Wilson interval (for bionamial) to rank comments.
* Domain consideration: time
    * Hacker News algorithm:
        * Slight damping effect for newer votes.

                numerator = (upvotes - downvotes - 1) ** DECAY)  # Net upvotes minus submitters upvote, with a slight decay (0.8) to damp later votes.
                denomiator = (time_now - time_posts) ** GRAVITY  # Gravity is used to ensure newer posts are higher, and older posts are less weighed by time (1.8).
                numerator / denomiator * PENALTY TERM  # Penalty is used to implement business logic to enforce certain community standards for HN.

    * Reddit algorithm:
        * Strong damping effect: votes 11 through 100 have the same impact as the first 10 votes.

                net_upvotes = upvotes - downvotes
                score = log(max(1, net_upvotes), 10) + sign * (net_upvotes) * t_posted / 45000

* Ranking wrap:
    * Theorertically grounded approaches: confidence interval, damping.
    * Some sites use ad hoc methods: tune it until it looks right.
    * Lots of formulas have constants, can be service dependant.
    * Can manipulate for good / evil.
    * Build based on domain properties, goals.
* Predict with sophisticed score:
    * Be careful with "damping", if users can infer actual average, then might question validity of site.

## Demographics and Related Approaches

* Motivation:
    * Might not like popular music, but might like popular music for your cohort:
        * Age, gender, race, socio-economic status etc.
    * Including non-demographics that may be predictive.
* How:
    * Use of post codes could indicate socio-economic status (in some cases).
    * Explore where data correlates with demographics:
        * Scatterplots, correlations...
* If you find relevant demographics:
    * Step 1: break down summary stats by demographic:
        * Most popular item for women, men.
        * Could break down gender groups by age.
    * Step 2: consider multiple regression model:
        * Predict items based on demographic stats.
            * Linear regression for mult-valued (rating) data
            * Logistic regression for 0/1 (purchase) data
* Need to deal with unknown demographics:
    * Overall preferences.
    * Demographics of newcomers
    * May be modeled separately.
* Getting demographics data is key:
    * Advertising networks, loyalty card sign ups.
    * Demographics can be inferred from data in some cases.
* Power and limits of demographics:
    * Work because products or content is created to reach them:
         * Tele programs.
         * Magazine articles and advertisements.
         * Personal products.
    * Products simply naturally appeal to different groups.
    * Demographics fail for people whose tastes don't match their demographic.

## Product Association Recommenders

* Ephemeral, contextual personalisation:
    * Personalised to what the user is doing right then:
        * Current navigation.
        * Doesn't known about long-term knowledge of prefs.
* How to compute?
    * Manually cross-sell tables:
        * If sales people reckon someone is buying a HDMI TV, sell them a HDMI cable too :)
    * Version 2: data mining associations
        * Maybe look for stuff likely to be bought in context.
* Data mining associations:
    * Simple first cut: % of X-buyers who also bought Y:

          X and Y
          -----
            X
            
        * Intuitively right, what what if X is top hats and Y is toilet paper?
        * Y may just be a thing a lot of people buy.
    * Next attempt: use Bayes Law to factor in the popularity of Y.

          P(toilet paper|top hats) = P(top hats|toilet paper) P(top hats)
                                     ------------------------------------
                                     P(toilet paper)

      * Then look at how much more likely toilet paper is than before :

            P(toilet paper | top hats)
            --------------------------
            P(toilet paper)

        * If ratio is close to 1, then top hats doesn't really change much.
    * Other solutions:
        * "Association rule mining" gives you lift metric:

              P(X and Y)
              -----------
              P(X) * P(Y)

            * Looks at non-directional association.
            * Looks at baskets of products, not just individual.
* Associate rules in practise:
    * Link recommendations: can I use the place someone came from to find recommendations.
    * Good recommender systems can make predicts the defy common sense (eg someone may be buying a product way less expensive than one recommended, and the recommended be a good idea).
    * Are all recommendations worth making?
        * Business rules to consider:
            * Do I actually have the product?
            * Offensive recommendations.
* Take aways:
    * Product association recommenders use current context to help recommend: where the user came from, products looked at.
    * Can figure out products to recommend from other users transactions. Need to balance high-probability with increase prob.
    * Can target to up sell or to sell different products.
