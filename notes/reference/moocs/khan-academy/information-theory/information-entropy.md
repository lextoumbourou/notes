---
title: Information Entropy (Information Theory)
date: 2021-07-28 00:00
status: draft
---

* If you had 2 machines that output an alphabet, with 2 different probability distributions, which one produces more information?
    * Machine 1 generates randomly:
        * P(A) = 0.25
        * P(B) = 0.25
        * P(C) = 0.25
        * P(D) = 0.25
    * Machine 2 generates accroding to probability:
        * P(A) = 0.5
        * P(B) = 0.125
        * P(C) = 0.125
        * P(D) = 0.25
 * Claude Shannon rephrased the question: what is the mininum yes or no questions you'd expect to ask to predict the next symbol?
 * You might start by asking a question that divides the space in half (is it A or B?).
     * Now you can eliminate half the possibilities.
 * Uncertainy of Machine 1 is 2 questions.
 * Uncertainy of Machine 2 is 1.75 questions on average. Since by asking is it A, 50% of the time it's only 1 question.
 * We say that Machine 2 is producing less information, because there's less "uncertainy" or surprise*
* Claude Shannon calls this measure of average uncertainy: [Information Entropy](../../../../permanent/information-entropy.md) and uses the letter `H` to represent.
* Unit of entropy is based on uncertainty of fair coin flip.
    * Called the bit
* Equation is $H=\sum\limits_{i=1}^{n} p_i \log_2 (1/p)$ which can be rewritten as $H=-\sum\limits_{i=1}^{n} p_i \log_2 (p)$
* Entropy is max when all outcomes are equally likely
* When predictability is introduced, entropy decreases.
