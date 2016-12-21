# Week 1

## Lesson 1: Probability

### Rules of Probability

* Rules defined as "events": event is some outcome that is observed (eg rolling heads or tails).
* Often describe event with a capital letter in math notation, eg: ``A``
* If event has 1/6 probability, you write: $$ P(A) = 1/6 $$.
* Probabilities must be between 0 and 1: $$ 0 \le P(A) \le 1 $$.
* Probabilities must add up to 1.
* Compliment of an event $$ A^c $$ means the event didn't happen.
	* Since prob adds to one: $$ P(A^c) = 1 - P(A) $$
* If A and B are 2 events, probability that A or B happens is probability of the union of the 2 events:
    
    $$ P(A \cup B) = P(A) + P(B) - P(A \cap B) $$ 

### Odds

* Odds are define as $$ O(A) = P(A) / P(A^c) $$
* The odds of you rolling a size are: $$ O(A) = (1/6) / (5/6) $$ or $$ 1/5 $$

### Expected value

* Probability times the value for each of the possible outcomes. eg $10 for heads, $-4 for tails = $$ 1/2 * 10 + 1/2 * -4 = $3 $$

### Classical and frequentist probability

* 3 frameworks you can use to define probability:
	* Classical
		* Only works well when probabilities are all equally likely.
	* Frequentist
		* Think about hypothetical sequence and figure out how often event occurs.  
	* Bayesian

### Bayesian probability and coherence

* Bayesian = "personal perspective".
* Leads to more intuitive results than frequentist approach.
* Can quantify using the concept of fair bet:
	* If rain, win $4. If no rain, lose $1. 4:1 odds. You know it's a fair bet, because expected value is 0.
	* Only balances out if probabilities match the payout. 
	* If you keep playing, will you eventually make money or break even?

## Discussion

Frequentist paradigm considered objective because it considers probability with actual, quantifiable information.

Bayesian considered objective because for most real-life events, lots of variables need to be considered before defining probabilities.

Frequentist considered subjective because most real life events are difficult to sample and some are hypothetical and cannot be sampled.

Bayesian subjective because it relies on potentially unreliable judgment to come up with the variables that define the probability of an event.

## Lesson 2: Bayes' theorem

### Conditional probability

* "Consider 2 events related to each other"
* What is prob of event A, given we know B happened:
    $$ P(A | B) = \frac{P(A \cup B)}{P(B)} $$
    * "Probability of A given B is probability of A and B over probability of B"
* Events are consider independent if: $$ P(A | B) = P(A) $$.
	* We can then infer: $$ P(A \cap B) = P(A)P(B) $$ (probability of A and B happening is probability of A times probability of B.

### Bayes' theorem

* Used to "reverse direction of conditioning"
	$$ P(A | B) = \frac{P(B | A)P(A)}{P(B|A)P(A) + P(B|A^c)P(A^c)} $$ 
    * Can be simplified to $$ \frac{P(A \cap B)}{P(B)} $$
* When 3 possible events, Bayesian theory expands to: $$ P(A_1 | B) = \frac{P(B | A_1)P(A_1)}{P(B | A_1)P(A_1) + P(B | A_2)P(A_2) + P(B | A_3)P(A_3)} $$