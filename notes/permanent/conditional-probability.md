---
title: Conditional Probability
date: 2024-08-11 00:00
modified: 2026-10-01 06:38
status: hidden
summary: Conditional probability, the general multiplication rule, total enumeration and Bayes' rule, explained through spam emails and dice rolls.
tags:
- Statistics
- Probability
---

**Conditional probability** is concerned with calculating the [Probability](probability.md) of an event, given that another event has occurred.

For example, you might expect the word "viagra" to appear more often in spam emails than in non-spam emails, sometimes called "ham". Let's use these illustrative probabilities:

$$P(\text{"viagra" in email} \mid \text{spam}) = 11\%$$

$$P(\text{"viagra" in email} \mid \text{ham}) = 0.01\%$$

The vertical bar $\mid$ means **given**. The first expression says that, given an email is spam, the probability that it contains the word "viagra" is $11\%$.

More generally, we write:

$$P(B \mid A)$$

This means the probability of event $B$, given that event $A$ has occurred. The order matters: the probability that a spam email contains a word is a different question from the probability that an email containing that word is spam.

## Calculating Conditional Probability

To compute conditional probability, we use:

$$P(B \mid A) = \frac{P(A \text{ and } B)}{P(A)}$$

This requires $P(A) > 0$, since we cannot divide by zero.

The denominator is the probability of $A$. The numerator is the probability that both $A$ and $B$ occur. Their ratio tells us how likely $B$ is within the cases where $A$ occurs.

## General Multiplication Rule

We can rearrange the conditional probability formula to get the **general multiplication rule**:

$$P(A \text{ and } B) = P(A) \times P(B \mid A)$$

To find the probability of both events, multiply the probability of $A$ by the probability of $B$ given $A$.

### Independent Events

In the special case where $A$ and $B$ are independent, knowing that $A$ occurred does not change the probability of $B$:

$$P(B \mid A) = P(B)$$

So the general rule reduces to the [Multiplication Rule of Probability](multiplication-rule-of-probability.md):

$$P(A \text{ and } B) = P(A) \times P(B)$$

For example, suppose we roll a fair six-sided die twice, with the rolls independent of each other. Knowing that the first roll was a five does not change the probability of getting a six on the second roll:

$$P(\text{second roll is 6} \mid \text{first roll is 5}) = P(\text{second roll is 6}) = \frac{1}{6}$$

## Computing Probability by Total Enumeration

Returning to the email example, suppose $20\%$ of emails are spam. Every email in this example is classified as either spam or ham, so the remaining $80\%$ are ham.

Let's use $S$ for the event that an email is spam, $H$ for ham, and $V$ for the event that it contains the word "viagra". We have:

$$P(S) = 0.20, \qquad P(H) = 1 - 0.20 = 0.80$$

$$P(V \mid S) = 0.11, \qquad P(V \mid H) = 0.0001$$

What is the probability that a randomly selected email contains "viagra", without knowing whether it is spam or ham?

There are two ways this can happen: the email contains "viagra" **and** is spam, or it contains "viagra" **and** is ham. These cases cover every email containing the word and are mutually exclusive, so we can use the [Addition Rule of Probability](addition-rule-of-probability.md):

$$P(V) = P(V \text{ and } S) + P(V \text{ and } H)$$

Using the general multiplication rule for each term:

$$P(V) = P(V \mid S) \times P(S) + P(V \mid H) \times P(H)$$

Substituting our probabilities:

$$P(V) = 0.11 \times 0.20 + 0.0001 \times 0.80$$

$$P(V) = 0.022 + 0.00008 = 0.02208 = 2.208\%$$

We can also see this by imagining 100,000 emails with exactly these proportions:

| Email type | Total emails | Emails containing "viagra" |
| --- | ---: | ---: |
| Spam | 20,000 | 2,200 |
| Ham | 80,000 | 8 |
| Total | 100,000 | 2,208 |

That gives $\frac{2208}{100000} = 2.208\%$.

This calculation is an example of the **law of total probability**: split the possibilities into mutually exclusive cases that cover all outcomes, calculate each case's contribution, and add them together. Each conditional probability is weighted by how likely its case is.

## [Bayes Theorem](bayes-theorem.md)

For a spam filter, we want to know the probability that an email is spam, given that it contains the word "viagra". So far, we have the probability in the other direction: how likely the word is to appear, given that the email is spam.

We can derive **Bayes' rule** from the conditional probability formula:

$$P(B \mid A) = \frac{P(A \text{ and } B)}{P(A)}$$

The order of events inside "and" does not matter. This is the **commutative law for conjunction**, one of the [Laws of Logic](laws-of-logic.md):

$$P(A \text{ and } B) = P(B \text{ and } A)$$

So we can rewrite the numerator, then apply the general multiplication rule:

$$P(B \mid A) = \frac{P(B \text{ and } A)}{P(A)} = \frac{P(A \mid B) \times P(B)}{P(A)}$$

This is Bayes' rule. As with the original conditional probability formula, it requires $P(A) > 0$.

### Applying Bayes' Rule to Spam

Using $S$ for spam and $V$ for the email containing "viagra":

$$P(S \mid V) = \frac{P(V \mid S) \times P(S)}{P(V)}$$

We already calculated $P(V) = 0.02208$, so:

$$P(S \mid V) = \frac{0.11 \times 0.20}{0.02208} \approx 0.99638 \approx 99.6\%$$

Under our illustrative assumptions, an email containing "viagra" has about a $99.6\%$ probability of being spam. We can check this against the table above: of the 2,208 emails containing the word, 2,200 are spam.

$$P(S \mid V) = \frac{2200}{2208} \approx 99.6\%$$

### Expanded Bayes' Rule

Sometimes the denominator, $P(A)$, is not given directly. We can calculate it using the law of total probability, splitting the possibilities into $B$ and its [Complement Rule](complement-rule.md), $B^c$, meaning "$B$ does not occur".

When both cases have nonzero probability:

$$P(A) = P(A \mid B) \times P(B) + P(A \mid B^c) \times P(B^c)$$

Substituting this into Bayes' rule gives the expanded form:

$$P(B \mid A) = \frac{P(A \mid B) \times P(B)}{P(A \mid B) \times P(B) + P(A \mid B^c) \times P(B^c)}$$

Here, $P(B^c) = 1 - P(B)$. In the spam example, the complement of spam is ham, so:

$$P(S \mid V) = \frac{0.11 \times 0.20}{0.11 \times 0.20 + 0.0001 \times 0.80} \approx 99.6\%$$

Use the shorter formula when the denominator is already known. If it isn't, calculate it separately by total enumeration or use the expanded formula. Both forms give the same result.
