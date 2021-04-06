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
    $$ P(A | B) = \frac{P(A \cap B)}{P(B)} $$
    * "Probability of A given B is probability of A and B over probability of B"
* Events are consider independent if: $$ P(A | B) = P(A) $$.
	* We can then infer: $$ P(A \cap B) = P(A)P(B) $$ (probability of A and B happening is probability of A times probability of B.

### Bayes' theorem

* Used to "reverse direction of conditioning"
	$$ P(A | B) = \frac{P(B | A)P(A)}{P(B|A)P(A) + P(B|A^c)P(A^c)} $$ 
    * Can be simplified to $$ \frac{P(A \cap B)}{P(B)} $$
* When 3 possible events, Bayesian theory expands to: $$ P(A_1 | B) = \frac{P(B | A_1)P(A_1)}{P(B | A_1)P(A_1) + P(B | A_2)P(A_2) + P(B | A_3)P(A_3)} $$
 
* ## Lesson 3: Bernoulli and binomial distributions

a

* Distribution when you only consider 2 potential outcomes (eg heads/tails)
* Use ``~`` to denote that a random variable follows a distribution:

    $$ X \sim B(p) \hspace{4ex} P(X=1) \\ \hspace{11ex}P(X=0) = 1-p $$

* Written as a function for all possible outcomes:

    $$ f(X=x | p) = f(x|p) = p^x(1-p)^{1-x}  \mathbb{1}_\{x \epsilon \{0, 1\}^(x)\} $$
* Referred to as "probability mass function" in math.

### Binomial distribution

* Distribution of multiple bernoulli distributions:
	$$  $$

### Uniform distribution

* A continuous random variable can be defined based on its "probability density function" or PDF.
* Key idea: integrate PDF over an interval, you get probability that the random variable is in that interval.

* Consider uniform distribution:
    
    $$ X \sim U[0, 1] $$
    * Can take any value between 0 and 1 and they're all equally likely.

    * Define PDF as follows: 

        $$ f(x) = \begin{cases} 1, & \text{if $x \in [0, 1]$}\\ 0, & \text{otherwise} \end{cases} $$

    * Can represent as indicator function:
        
        $$ I_{\{0 \le x \le 1\}}{}^{(x)} $$

* Probability is calculating by integrating the density function. In this example, calculating probability result is between 0 and 1/2.

    $$ P(0 < x < \frac{1}{2}) = \int_{0}^{\frac{1}{2}} f(x)dx = \int_{0}^{\frac{1}{2}}dx = \frac{1}{2} $$

#### Key rules for probability density functions

* Integral from - infinity to + infinity must add up to 1:

    $$ \int_{-\infty}^{\infty} f(x)dx = 1 $$

* Densities must be non-negative for all possible values of X:

    $$ f(x) \le 0 $$

* Expected value for continuous random variable is the integral of x * f(x) * d(x):

    $$ E[X] = \int_{-\infty}^{\infty} x f(x) dx $$

    * Expected value for a function (replace x with function):

        $$ E[g(x)] = \int_{-\infty}^{\infty} g(x) f(x) dx $$

* Expectation of some constant times a random variable X, constant times expected value of x:

    $$ E[cX] = cE[x] $$

* Expectation of the sum of 2 random variables is the sum of expectations:

    $$ E[x + y] = E[x] + E[y] $$

* If x and y are independent, then the expected values of the product of them is the product of the expectations:

    $$ \text{if x ⫫ y, then E[xy]} = E[x]E[y] $$

### Exponential and normal distributions

#### Exponential

* Another example of a continuous distribution.
* Used when events are occurring at a particular rate and exponential is the "waiting time" between events (don't understand this).
  
* Density function for a given lambda:

    $$ f(x | \lambda) = \lambda_e^{-\lambda x}\space \space\text{for x} \ge 0$$

* Expected value:

    $$ E[x] = \frac{1}{\lambda} $$

* Variance:

    $$ var(x) = \frac{1}{\lambda^2} $$

### Uniform distribution

* Density function:

    $$ f(x | \theta_1, \theta_2) = \frac{1}{\theta_2 - \theta_1} I_\{\theta_1 \le x \le \theta_2\}(x) $$

### Normal distribution (aka guassian distribution)

* X follows the normal distribution with mean mu and variance sigma squared:
    $$ X \sim N(\mu, \sigma^2) $$
* Probability density function:
    $$ f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\{-\frac{1}{2\sigma^2}(x-m)^2\} $$
