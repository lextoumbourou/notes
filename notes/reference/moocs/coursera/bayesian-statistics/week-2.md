# Week 2

## Frequentist inference

### Background

#### Products and Exponents

* Product notation: $$\prod $$
* Can redefine factorial $$n! $$as $$\prod_{i=1}^{n} \text{i for n} \le 1 $$
* Exponents form: $$a^x $$where a is called the base and x is the exponent.
* Exponents have the following useful properties:
  * $$a^x * a^y = a^{x + y} $$
    * Note: base must be the same for this to work. 
  * $$(a^x)^y = a^{x * y} $$

#### Natural Logarithm

* Logarithms are the inverse of exponential function.
* If $$y = a^x $$then $$\log_a(y) = x $$
* The "natural logarithm" has base e and can be written without the subscript e:
  * $$\log_e(y) = \log(y) $$
* $$log(y) $$is only defined for $$y > 0 $$
* $$\exp(\log(y)) = \log(\exp(y)) = y $$
* Logarithm has the following properties:
  * $$\log(x * y) = \log(x) + \log(y) $$
  * $$\log(\frac{x}{y}) = \log(x) - \log(y)  $$
  * $$\log(x^b) = b\log(x) $$
  * $$\log(1) = 0 $$
* Since natural logarithm is a "monotonically increasing" (aka it always increases) "one-to-one" (aka 'if f(a) = f(b) implies that a = b then f is one-to-one') function.
* Finding that x that maximises any $$f(x) $$is the same as maximising $$\log(f(x)) $$

#### Argmax

* When maximising a function, you are interested in 2 things:
  1. The value $$f(x) $$can get to when maxed, written as $$\max_x f(x) $$
  2. The x-value that maximises the func, written as $$\hat{x} = \arg \max_xf(x) $$

### Confidence intervals

* Frequentist approach to inference: view data as random sample from larger, hypothetical population.
* Example, coin flip result: 44 heads, 56 tails.
  * Can be viewed as sample from much larger population.
  * Can say that each flip follows the a Bournelli distribution with prob p: $$X_i ~ B(p) $$
  * Can ask: "What's the prob of getting heads?"
  * Can also ask: "how confident are we in that estimate?"
  * By Central Limit Theorum:
    * $$\sum\limits_{i=1}^{100} X_i \sim N(100p, 100p(1-p)) $$
    * Can say: $$100p - 1.96 \sqrt{100p(1-p)} and 100p + 1.96 \sqrt{100p(1-p)} $$
    * Since we observed 44 heads, for that sample $$\hat{p} = \frac{44}{100} $$
    * Confidence interval calculated as: $$44 \pm 1.96\sqrt{44(.56)} = 44 \pm 9.7 = (34.3, 53.7)$$
    * Can say "we're 95% confident the probability of heads falls between .343 and .537".

### Likelihood function and maximum likelihood

