# Week 2: Nearest Neighbour Search

## Introduction to nearest neighbor search and algorithms

## Retrieval as k-nearest neighbour search

* Document retrieval
  * Goal: retrieve an article that might be relevant to user based on what they're currently reading.
  * Take article, find all articles that are similar to current article, then sort based on closeness to current article.

## 1-NN algorithm

* Input: Query article
         Corpus of documents (N docs)
* Output: Most similar article.
  * Formally: ``Xnn = min_distance(Xq, Xi)``
  * In code:

    ```
    def one_nn(corpus, query_doc):

      """One nearest neighbour."""

      closest_distance = float('inf'), closest_doc = None
      for i in range(len(corpus)):
        distance = distance(document[i], query_doc)
        if distance < closest_distance:
          closest_distance = distance
          closest_doc = document[i]

      return closest_doc
    ```

    (Obviously the interesting part is going to be the distance calculation).

## k-NN algorithm

* Same as 1-NN but return a set of documents that are closest to your document.
  * Use queue to shift further article when a closer one comes along.

## The importance of data representations and distance metrics

### Document representation

* Bag of words model:
  * Count number of instances of words
  * Ignore order
* Common words like ``the``, ``and`` can dominate word counts; enter TF-IDF
  * Emphasises words that are common locally but are rare globally.
  * Term frequency = word counts in document
  * Inverse doc freq: ``log(# docs / 1 + # docs using word)``
  * ``tf * idf``

### Distance metrics: Euclidean and scaled Euclidean

* In 1D, Euclidean distance: ``distance(x_i, x_q) = abs(x_i - x_q)``
  * Would only work if you had 1 word in your vocabulary.
* In multi dimensions:
  * many interesting distance functions
  * can weight dimensions differently
* Reasons for weighting different features:
  * Some features more relevant than others.
    * Example: weight the title more than the body
  * Some features vary more than others (??)
    * Specify weights as function of feature spread
      * for feature j: ``1 / max_i(x_i[j]) - min_i(x_i[j])``
* Scaled Euclidean distance:
     $$ distance(\mathbf{x}_i, \mathbf{x}_q) = \sqrt{a_1(\mathbf{x}_i[1] - \mathbf{x}_q[1])^2 + ... + a_d(\mathbf{x}_i[d] - \mathbf{x}_q[d])^2} $$
	* Add up the distance for each scaled feature, between the query doc and some other doc and take the square root.
	* Can turn off features by setting weight to 0.

### Writing (scaled) Euclidean distance using (weighted) inner products

* Can rewrite the non-scaled Euclidean distance using linear algebra. Example:
    
    * $$ distance(\mathbf{x}_i, \mathbf{x}_q) = \sqrt{(\mathbf{x}_i[1] - \mathbf{x}_q[1])^2 + ... + (\mathbf{x}_i[d] - \mathbf{x}_q[d])^2}  $$ can be rewritten as:
	* $$ distance(\mathbf{x}_i, \mathbf{x}_q) = \sqrt{(\mathbf{x}_i - \mathbf{x}_q)^T(\mathbf{x}_i - \mathbf{x}_q)} $$ 
	* Basically: taking the dot product of the two feature distance vectors is equivalent to squaring then adding the results.
* To add the feature weights, you can add a diagonal matrix of features (note: not clear on the diagonal matrix part):
	* $$ distance(\mathbf{x}_i, \mathbf{x}_q) = \sqrt{(\mathbf{x}_i - \mathbf{x}_q)^T\color{blue}{\mathbf{A}}(\mathbf{x}_i - \mathbf{x}_q)} $$


### Distance metrics: Cosine similarity

* Another inner product measure: multiple features in one document vs another then add up results:

	* $$ \sum\limits_{j=1}^{d}\mathbf{x}_i[j]\mathbf{x}_q[j] $$
	* Higher is better (map overlap). Lowest possible = 0.

* Cosine similarity: same as above, however, you divide the result by the normalised vector of features:
	* $$ \frac{\sum\limits_{j=1}^{d}\mathbf{x}_i[j]\mathbf{x}_q[j]}{\sqrt{\sum\limits_{j=1}^{d}(\mathbf{x}_i[j])}\sqrt{\sum\limits_{j=1}^{d}(\mathbf{x}_q[j])}} $$
	* Can be rewritten like: $$ \frac{\mathbf{x}_i^T\mathbf{x}_q}{||\mathbf{x}_i|| ||\mathbf{x}_q|| } $$
* Cosine distance = 1 - cosine similarity
	* Not a proper distance metric because "the triangle equality doesn't hold".
	* Efficient to compute for sparse vectors because you only need to compute non-zero elements.
	* Very similar documents would result in a score close to 1. $$ cos(0) \approx 1 $$
	* With only positive features, similarity would be between 0 and 1.

### Normalise or not?

* When you don't normalise, a document twice the size of the other would result in a much larger similarity score.
* Normalisation is not always desired: normalising can make dissimilar objects appear more similar, especially with smaller documents.
    * Compromise: cap maximum word counts.
* Other distance metrics: Mahalanobi, rank-based, correlation-based, Manhatten .
* Combining distance metrics:
	1. Text of document (cosine similarity)
	2. \# of reads of the doc (euclidean distance)
	3. Add together with user-specified weights    