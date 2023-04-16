---
title: "Week 4 - Clustering and Similarity: Retrieving Documents"
date: 2015-10-07 00:00
category: reference/moocs
status: draft
parent: ml-foundations
---

* Overview:
    * Have lots of observation and want to infer structure.
    * Structure in this module: groups of related observations or clusters
    * Task: retrieving documents of interest.
* Document retrieval task?
    * Reading article you like: want to retrieve similar articles.
    * Challenges:
        * How do we measure similarity?
        * How do we search over articles?
* Word count representation for measuring similarity
    * Bag of words model.
        * Ignores order of words.
        * Counts # instances of each word in vocabulary.
* Can measure similarities by multipling word count over two vectors with word counts at same position (element-wise multiplication) and add the results.

            word_pos = ['hello', 'world', 'what', 'is', 'up']
            word_count_1 = [1, 1, 0,  0,  0]
            word_count_2 = [2, 0, 0, 0, 0]
            similarity_score = sum(
                [word_count_1[pos] * word_count_2[pos]
                for pos in range(len(word_pos))])

* Problem: bias against larger documents.
    * Solution: *normalised count vector*.
        * Computing norm of vector: add square of every entry in the vector and take square root.
* Problem: Common words "dominate" rare words in similarity score and usually rare words are important to doc context..
    * Solution: prioritize important words with [TF-IDF](../../../../permanent/tf-idf.md)
        * tf-idf (term frequency - inverse document frequency).
        * Don't want to emphasis only rare words; want "important words".
            * "Common locally": appear frequently in document.
            * "Rare globally": appear rarely in corpus (rare globally).
        * Rough algorithm:
            * Count words in document ("local" term frequency).
            * Count inverse document freqency getting the ratio against whole corpus:

                      word_frequency_ratio = log(
                          number_of_documents / (1 + number_of_documents_term_appears_in))

            * ``+ 1`` to deal with terms that appear in no documents (avoid dividing by 0).
            * ``log`` to reduce the impact of result (I guess?)
        * Multiply inverse document frequency with term frequency (``tf * idf``).
* Retrieving similar documents
    * Nearest neighbor
        * Specify: distance measure (similarity calculated above, for example).
        * Ouput: set of most similar articles.
    * Examples
        * 1 - Nearest neighbor
        * Find the most similar.
        * K - Nearest neighbors
            * Keep priority queue of top K found articles.
* Clustering documents
    * Goal: discover groups (clusters) of related articles.
    * If labels are known: you could have a training set of labeled docs.
    * Simply a multiclass classification problem.
    * If not: unsupervised problem.
        * Input: word count vector for docs.
        * Output: cluster labels by grouping docs with similar word counts.
    * Cluster defined by center and shape/spread.
    * Assign observation (doc) to cluster (topic label).
* k-means
    * Specify number of clusters and find closest to an average point (hence ``K`` means).
    * Similarity metric == distance to cluster centre.
    * Overview:
        * Initialize cluster centres (initially just randomly).
        * Assign observations to closest cluster centre.
            * Uses "Voronoi Tessellation":
                * Look at cluster centres.
                * Define regions around them.
                * Put any points that fall in the region in clusters.
        * Revise cluster centres as mean of assigned observations.
            * Update definitions of cluster centres based on where stuff has been assigned.
            * Change "centre of masses for clusters".
            * Iterate until "convergence".
* Other examples of clustering
     * Image search - clustering similiar images.
     * Grouping patients by medical condition.
         * Grouping brain scans.
     * Grouping products on Amazon
     * Grouping related users.
     * Discovering similar neighbourhoods.
     * Forecast violent crimes to regions.
* Clustering and similarity ML block diagram
    * Training data: (doc id, doc text)
    * Extract features (tf-idf)
    * Put features through ML Model (clustering)
        * y-hat == estimated cluster label.
    * Minimize distances between cluster centres (k-means).
