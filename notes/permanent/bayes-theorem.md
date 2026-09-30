---
title: "Bayes Theorem"
date: 2024-08-12 00:00
modified: 2026-10-01 07:27
cover: /_media/bayes-theorem-blood-tests-daniel-sone.jpg
cover_credits: Photo by Daniel Sone / <a href="https://unsplash.com/@nci">National Cancer Institute</a> on <a href="https://unsplash.com/photos/clear-glass-test-tubes-egT3xtDu9DQ">Unsplash</a>, cropped.
aliases: Bayes Rule
summary: Update the probability of an event based on new evidence
tags:
- Statistics
- Probability
---

**Bayes' theorem** (or **Bayes' rule**) allows us to update the probability of an event based on new evidence.

It builds on [Conditional Probability](conditional-probability.md). The basic form is:

$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}$$

The **prior** is the probability before considering the new evidence. The **posterior** is the updated probability after considering it.

## Example: A Positive Disease Test

Let's say we want to compute the probability that a person has a disease, given that they test positive for it:

$$P(\text{Disease} \mid \text{Positive test})$$

We'll use these illustrative figures:

- $1\%$ of the population has the disease: $P(\text{Disease}) = 0.01$.
- The test returns a positive result for $90\%$ of people with the disease: $P(\text{Positive test} \mid \text{Disease}) = 0.90$.
- The test also returns a positive result for $15\%$ of people without the disease: $P(\text{Positive test} \mid \text{No disease}) = 0.15$.

The $1\%$ figure is our **prior**. The $90\%$ figure is the **likelihood** of a positive result given the disease. The $15\%$ figure is the **false-positive rate**.

Using Bayes' theorem:

$$P(\text{Disease} \mid \text{Positive test}) = \frac{P(\text{Positive test} \mid \text{Disease}) \times P(\text{Disease})}{P(\text{Positive test})}$$

The denominator is the probability of getting a positive test result overall, whether or not the person has the disease. This is what **evidence** means in the formula.

### Calculating the Probability of a Positive Test

The only value we don't yet know is $P(\text{Positive test})$.

A positive result can come from someone with the disease or someone without it. These cases are mutually exclusive and cover all positive results, so we can use the [Addition Rule of Probability](addition-rule-of-probability.md):

$$
\begin{aligned}
P(\text{Positive test})
&= P(\text{Positive test and Disease}) \\
&\quad + P(\text{Positive test and No disease})
\end{aligned}
$$

Then we apply the general multiplication rule to each term:

$$
\begin{aligned}
P(\text{Positive test})
&= P(\text{Positive test} \mid \text{Disease}) \times P(\text{Disease}) \\
&\quad + P(\text{Positive test} \mid \text{No disease}) \times P(\text{No disease})
\end{aligned}
$$

This is the **law of total probability**, which we used for total enumeration in [Conditional Probability](conditional-probability.md).

Since $P(\text{No disease}) = 1 - 0.01 = 0.99$, substituting our values gives:

$$P(\text{Positive test}) = 0.90 \times 0.01 + 0.15 \times 0.99$$

$$P(\text{Positive test}) = 0.009 + 0.1485 = 0.1575 = 15.75\%$$

### Updating the Probability of Disease

Now we can fill in the denominator:

$$P(\text{Disease} \mid \text{Positive test}) = \frac{0.90 \times 0.01}{0.1575} \approx 0.05714 \approx 5.7\%$$

Under these assumptions, a positive result raises the probability of having the disease from $1\%$ to about $5.7\%$.

The result is still relatively low because the disease is rare and false positives are common. In an imaginary population of 10,000 people with exactly these proportions, 90 people with the disease and 1,485 people without it would test positive. Only 90 of the 1,575 positive results would be from people with the disease.

## General Form

For general events $A$ and $B$, Bayes' theorem is:

$$P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}$$

This requires $P(B) > 0$. If $P(B)$ is not already known, we can expand it using the law of total probability:

$$P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B \mid A) \times P(A) + P(B \mid A^c) \times P(A^c)}$$

Here, $A^c$ is the [Complement Rule](complement-rule.md) of $A$, meaning that $A$ does not occur, and $P(A^c) = 1 - P(A)$. The expanded form assumes both cases have nonzero probability so their conditional probabilities are defined.

Use the shorter formula when the denominator is already known. Otherwise, calculate it separately or use the expanded form. Both approaches give the same result.
