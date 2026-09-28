---
title: Probability
date: 2023-09-02 00:00
modified: 2026-09-29 08:20
cover: /_media/probability-dice-richard-heinen.jpg
cover_credits: Photo by <a href="https://unsplash.com/@richard_he">Richard Heinen</a> on <a href="https://unsplash.com/photos/a-pile-of-colorful-dices-stacked-on-top-of-each-other-o9Sw50EVhzU">Unsplash</a>, cropped.
summary: "How likely an event is to occur"
tags:
- Statistics
---

The **probability** of an event describes how likely it is to occur.

The convention for describing the probability of an event $A$ is to write it as $P(A)$.

When there is a finite number of equally likely outcomes, the probability of an event is the proportion of those outcomes in which it occurs.

$$P(A) = \frac{\text{Number of outcomes in }A}{\text{Total number of possible outcomes}}$$

For example, a standard deck has 52 cards. The probability of drawing the ace of spades from a randomly shuffled deck is:

$$P(\text{ace of spades}) = \frac{1}{52} \approx 1.92\%$$

A fair coin has two equally likely outcomes, heads and tails, so:

$$P(\text{heads}) = \frac{1}{2} = 50\%$$

We can also estimate probability from observed data:

$$\widehat{P}(A) = \frac{\text{Number of times }A\text{ occurred}}{\text{Number of trials}}$$

The hat in $\widehat{P}(A)$ indicates an estimate.

For example, the CDC reports 1,045 male births per 1,000 female births in the USA in 2024 (Table A) [@ostermanBirthsFinalData2026]. Dividing male births by total births gives:

$$\widehat{P}(\text{baby is a boy}) = \frac{1045}{1045 + 1000} \approx 51.1\%$$

This observed proportion can be used to estimate the probability that a future live birth in the USA will be a boy.

There's also a type of probability you'll often hear about day to day that expresses how strongly someone believes an event will occur. These estimates can draw on evidence, experience or just gut feel, without being calculated directly from observed outcomes across repeated experiments. This type of probability is called **subjective probability**.

For example, there's a lot of talk about the likelihood of AI causing human extinction or a similarly severe catastrophe, and many different public figures in AI have shared their estimates of [P(doom)](https://en.wikipedia.org/wiki/P%28doom%29).

Probabilities are always between 0 and 1, inclusive. Or in formal language:

$$0 \leq P(A) \leq 1$$

## 4 Basic Rules of Probability

### [Complement Rule](complement-rule.md)

The probability of event $A$ **not** occurring is $1-P(A)$. The event that $A$ does not occur is called its **complement**.

$$P(\text{A does not occur}) = 1 - P(A)$$

For example, let $A$ be the event of drawing the ace of spades from a randomly shuffled standard deck:

$$P(A) = \frac{1}{52} \approx 1.92\%$$

Then the probability of drawing any other card is:

$$P(\text{not the ace of spades}) = 1 - \frac{1}{52} = \frac{51}{52} \approx 98.08\%$$

### [Rule for Equally Likely Outcomes](rule-for-equally-likely-outcomes.md)

If there are $n$ possible outcomes and they are all equally likely, each outcome has probability $\frac{1}{n}$. An event can include more than one outcome, so:

$$P(A) = \frac{\text{Number of outcomes in }A}{n}$$

Sticking with the cards, there are 13 spades in a standard deck. The probability of drawing any spade is:

$$P(\text{drawing a spade}) = \frac{13}{52} = 25\%$$

### [Addition Rule of Probability](addition-rule-of-probability.md)

Two events are **mutually exclusive** if they cannot both occur.

For example, in Mario Kart, driving through an item box gives you an item.

![A colourful, transparent Mario Kart 8 item box with a question mark](../_media/mario-kart-8-item-box.png)

*Mario Kart 8 item box. Artwork © Nintendo [@nintendoMarioKart8Items].*

To keep the maths simple, let's give our example box three possible outcomes and made-up probabilities. It gives exactly one item:

- A banana: $40\%$.
- A shell: $35\%$.
- A mushroom: $25\%$.

Let $A$ be getting a shell and $B$ be getting a mushroom. These events are mutually exclusive: our box cannot give you both in the same pickup.

For mutually exclusive events, we can add their probabilities to find the probability of either occurring:

$$P(A \text{ or } B) = P(A) + P(B)$$

So the probability of getting either a shell or a mushroom is:

$$P(A \text{ or } B) = 0.35 + 0.25 = 0.60 = 60\%$$

There is no overlap to count twice, because each pickup gives exactly one of these items.

### [Multiplication Rule of Probability](multiplication-rule-of-probability.md)

Two events are **independent** if knowing whether one occurred does not change the probability of the other.

For independent events, we multiply their probabilities to find the probability that both occur:

$$P(A \text{ and } B) = P(A) \times P(B)$$

For example, suppose we roll a fair six-sided die twice, with the rolls independent of each other. The first result does not change the probability of rolling a six on the second roll:

$$P(\text{six on both rolls}) = \frac{1}{6} \times \frac{1}{6} = \frac{1}{36} \approx 2.78\%$$

### Example: At Least One Six in Three Rolls

What is the probability of getting at least one six in three independent rolls of a fair six-sided die?

"At least one" includes getting a six once, twice or on all three rolls. It's easier to calculate the probability of the opposite event, getting no sixes, and use the complement rule:

$$P(\text{at least one six}) = 1 - P(\text{no sixes})$$

On each roll, five of the six outcomes are not a six, so the probability of not rolling a six is $\frac{5}{6}$. Getting no sixes means not rolling a six on the first roll **and** the second roll **and** the third roll. Since the rolls are independent, we multiply:

$$P(\text{no sixes in three rolls}) = \frac{5}{6} \times \frac{5}{6} \times \frac{5}{6} = \left(\frac{5}{6}\right)^3 = \frac{125}{216}$$

Then subtract from 1:

$$P(\text{at least one six in three rolls}) = 1 - \frac{125}{216} = \frac{91}{216} \approx 42.13\%$$

Simply adding $\frac{1}{6}$ three times would not work here. Getting a six on different rolls is not mutually exclusive: a sequence such as $(6, 6, 2)$ would be counted twice.