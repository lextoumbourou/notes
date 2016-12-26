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
