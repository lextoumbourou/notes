# Location Tagging in Text

Author: Shawn Brunsting

## Algorithm

### 2.2 Terminology

* **Phrase**: word or sequence of adjacent words in the text.
    * Store as a single string.
    * $$ P $$ is set of unique phrases considered a possible location.
* **Term** is object associated with phrase, include metadata like position of phrase in text.
	* Multiple of the same phrase only have 1 term. Therefore: $$ |P| \le |T| $$.
	* For some $$ t \in T $$ let $$ t_{phr} $$
* ** Result **: single location in knowledge base. For each phrase $$ p \in P $$ there's a set of results $$ R^p $$
	* For each result $$ r \in R^p $$ the name of result is similar to the phrase. Let $$ R_t = R^{t_{phr}} \forall t \in T  $$ 
* For a term $$ t \in T $$ and a result $$ r \in R^t $$ define the score to be $$ S^t_r $$. Score is used to disambiguate.
* For 2 terms, $$ t_1, t_2 \in T $$ define $$ W^{t_1}_{t_2} $$ to be weight of term $$ t_2 $$ when we assume $$ t_1 $$ is a true reference.
	* Weights are used to reduce bias when terms conflict with each other. 

#### Technical overview

1. For all terms that are potential location referencing using part-of-speech tagging and named entity recognition using sets $$ T $$ and $$ P $$.
2. For each phrase $$ p \in P $$, query knowledge base with $$ p $$ and save results to $$ R^p $$
3. Reduce set $$ T $$ to remove conflicting terms. Update set $$ P $$ to reflect changes in $$ T $$. For each phrase $$ p \in P $$ match $$ p $$ to a single result $$ r \in R^p $$ using distance-based scores to each results, and getting results with best scores.

### 2.3 Location Extraction

#### Tagging text

* Extract tags text using a Stanford part-of-speech (POS) tagger and a named entity recognizer (NER).
  * Uses both taggers because both are imperfect: one makes up for faults in the other.
* POS assigns tags to words. The tags used are:

  * ``CC`` - coordinating conjunction - grouped as conjunction.
  * ``CD`` - cardinal number - grouped as adjective.
  * ``IN`` - preposition or subordinating conjunction - grouped as preposition.
  * ``JJ`` - adjective - grouped as adjective.
  * ``NN`` - noun, singular or mass - grouped as noun.
  * ``NNS`` - noun, plural - grouped as noun.
  * ``NNP`` - proper noun, singular - grouped as noun.
  * ``NNPS`` proper noun, plural - grouped as noun.
  * ``TO`` - to - grouped as preposition

* NER tags words with 4 possibilities: ``LOCATION``, ``PERSON``, ``ORGANIZATION`` or ``O`` (other)

#### Extracting terms

* Any phrase in text is considered potential location if:

  1. Phrase contains at least one word with a noun tag from POS tager or LOCATION tag from NER.
  2. All words in phrase are tagged with noun or adjective tag from POS choices or a LOCATION tag from NER.

* Location like ``New York`` have multi word names. NER and POS tags words individually so we need to consider multi-word possibilities when building $$ T $$.
* If multiple nouns appear next to each other, computer doesn't know if multi locations or one location, so add all to $$ T $$ and resolve conflicts.
  * Example: phrase "New York City" would result in six terms added to T: "new", "york", "city", "new york", "york city" and "new york city".
* Adjectives that are part of multi-word phrases with other nouns are considered (only adjectives are not).
  * "Georgian college" would result in "college" and "georgian college" added to T.
* Numbers are treated like adjectives too, to account for street addresses: "200 University Avenue" would generate: "University", "Avenue", "University Avenue", "200 University" and "200 University Avenue".

#### Removing Terms

* Reduce T by finding terms that contain words tagged as locations by NER model and discarding all the rest.
* If no words tagged with LOCATION, then algo relies solely on the results of POS tagger.
  * Look for terms after prepositions: "Bob travelled from Waterloo" contains preposition "from", which tells us Waterloo is location and Bob travelled from there.
  * Attempt to ignore "temporal prepositions" by ignoring word "for" as preposition (any other temporal prepositions?)
  * If text contains terms after prepositions, all other words are discarded.
  * Term considered a preposition if all words between prepos and term and tagged with any tags in 2.1 including conjuctions (so "the people travelled from New York and Toronto") would be consider both locations.
* Where text contains no LOCATION tags or no terms following prepositions, no terms are removed from T.

### 2.4 Searching a Knowledge Base

* Query Nominatim: https://nominatim.openstreetmap.org and return top n results (10 by default).

### 2.5 Disambiguation

#### 2.5.1 Distance

* Key to disambiguation is physical distance between results.
* If we have location mentions that are ambiguous, try to resolve using distance between other location mentions:
  * Example: "I live in Melbourne, Australia".
    * Melbourne is closer to Australia than Florida, so it wins.
* Uses "great circle distance" using the mean radius of earth (6371):

  ```
  a_x, a_y = result_1.long, result_1.lat
  b_x, b_y = result_2.long, result_2.lat
  d(a, b) = 6371 * arccos(sin(a_y) * sin(b_y) + cos(a_x) * cos(b_x) * cos(a_x - b_x))
  ```

#### 2.5.2 Weights

* Weights used when terms conflict.
* Example of conflicting terms: "York" and "New York".
  * Text could not have referred to a location called York and another called New York.
* Assigning weights lets you reduce bias in the disambiguation step, until we can determine which interpretation is correct.
* Weight: $$ W_{t_2}^{t_1} $$ probability that t2 is a true location, given t1 is a true location.
* Weights are heuristically defined: not true probabilities, but have many of the same properties as probabilities.
* Weights between 2 points are always <= to 1 for all pairs in the set of terms.
    * If no terms conflict, than the weights are all 1.
* In the conflicting case, the weight is set to 0.
    $$ W_{t_2}^{t_1}=0 \space\space \forall t_1, t_2 \in T|t_1 \space\text{and}\space t_2\space \text{conflict} $$
* A term does not conflict with itself:
    $$ W_{t}^{t}=1 \space\space \forall t \in T $$
* Conflict terms end up in groups. All conflicting terms should be added to same group. If $$ G(t) $$ is a group for term t then:
    $$ G(t_1) = G(t_2) \space\space \forall t_1, t_2 \in T|t_1 \space\text{and}\space t_2\space \text{conflict} $$
* Therefore, if a single result doesn't conflict with any others, it should be in a group of size 1.
    $$ G(t_1) = \left\{t1\right\} \space\space \forall t_1 \in T|t_1 \space\text{does not conflict with any } T $$
* An "interpretation" is a subset of a group with no conflicts.
* Define the weight for $$ W_{t_2}^{t_1} $$ when t1 does not overlap with t2, as follows:
    $$ W_{t_2}^{t_1} = \frac{1}{\text{# of interp. of } G(t_2)} \sum\limits_{\text{interp. of G(t2) that contain t2}}\frac{1}{\text{# of terms in this interp.}} $$
    * Basically, the weights turn out to be much higher for phrases that exist alone in a group with no other interpretations (eg New York City ends up with a weight of 1/4 where New ends up with a weight 5/24)
* Lastly, need to handle the weight for $$ W_{t_2}^{t_1} $$ in the case where t1 in the t2 set and t1 and t2 don't conflict: to do that, temporarily remove any terms from the t2 group that conflict with t1.

#### 2.5.3 Scoring functions

* For 2 search results, r1 and r2, d(r1, r2) is the distance between them.
* Shorthand for shortest distance between a result and any result for a set of results for a phrase: $$ c(r, t) = min_{r^1 \in R^t} d(r, r^1)  $$
* Paper defines 8 scoring functions:
    * Total Distance
    * Weighted Distance
    * Inverse
    * Weighted Inverse
    * Weighted Normalized Inverse
    * Inverse Frequency 
    * Weighted Inverse Frequency
    * Weighted Normalized Inverse Frequency

##### Total Distance

* Adds distance between a result and closest result for all others term:

    $$ S^{t^1}_{r^1} = -\sum\limits_{t^2 \in T \space\space W_{t^2}^{t^1} \ne 0} c(r^1, t^2) $$

* May be sensitive to locations that are far away.

##### Weighted Distance

* Same as total distance, but multiples the closest result by the weight:

    $$ S^{t^1}_{r^1} = - \sum\limits_{t^2 \in T \space\space W_{t^2}^{t^1} \ne 0} W^{t^1}_{t^2} c(r_1, t^2) $$

* May be sensitive to locations that are far away.

##### Inverse and Weighted Inverse

* Takes the reciprocal of the total distance and weighted distance.
* Handle divide by zero errors by taking the maximum of c and 10^-3

##### Weighted Normalized Inverse

* Normalizes all scores to ensure they're comparable to each other: one score divided by the total of all scores.

#### Disambiguation Algorithms

* 2 versions of disambiguation: 1-phase and 2-phase.

##### 1-phase

While there are terms to be disambiguated (if any has more than 1 result or if it conflicts with another term):
  Calculate the scores for each result.
  Find the best scoring result for for each term.
  Throw out any terms that have a weight of 0 compared to other terms.
 Recalculate weights

#### 2-phase

Same as 1-phase, except attempts to resolve all conflicts before performing disambiguation.