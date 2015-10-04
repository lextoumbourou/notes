# Week 4

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

    ```
    word_pos = ['hello', 'world', 'what', 'is', 'up']
    word_count_1 = [1, 1, 0,  0,  0]
    word_count_2 = [2, 0, 0, 0, 0]
    similarity_score = sum(
        [word_count_1[pos] * word_count_2[pos]
        for pos in range(len(word_pos))])
    ```

  * Problem: bias against larger documents.
  * Solution: normalise vector.

    * Computing norm of vector: add square of every entry in the vector and take square root.

  * Problem: Common words "dominate" rare words in similarity score and usually rare words are important to doc context..
  * Solution: prioritize important words with tf-idf.

* tf-idf (term frequency inverse document frequency).
  
  * Don't want to emphasis only rare words; want "important words".
    * "Common locally": appear frequently in document.
    * "Rare globally": appear rarely in corpus (rare globally). 
