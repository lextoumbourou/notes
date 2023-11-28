---
title: Week 19 - Combinatorics A
date: 2022-02-18 00:00
category: reference/moocs
status: draft
parent: uol-discrete-mathematics
modified: 2023-04-08 00:00
---

## Lesson 10.1 The basics of Combinatorics

* [Combinatorics](../../../../permanent/combinatorics.md)
    * The math topic that studies "finite" countable discrete structures: collections or arrangements of objects.
    * Involves counting objects and studying the mathematical properties of different arrangement of objects.
    * Applications include programming, physics, economics and other fields like prob theory.

## Lesson 10.103 The basics of counting

* [Product Rule](../../../../permanent/product-rule.md)
    * To determine the number of different possible outcomes in a complex process, we can break the problem into a sequence of two independent tasks:
        * if there are $n$ ways of doing the first task
        * for each of these ways of doing the first task, there are $m$ ways of doing the 2nd task.
        * then there are n * m different ways of doing the whole process.
    * Example
        * Problem
            * Consider a restaurant offering a combination meal where a person can order from each of the following categories:
                * 2 salads
                * 3 main dishes
                * 4 side dishes
                * 3 desserts
            * How many combinations meals are there to choose from?
        * Solution
            * The problem can be broken down into 4 independent events:
                * selecting a salad
                * selecting a main dish
                * selecting a side dish
                * selecting a dessert
            * For each event, the number of available options is:
                * 2 for the first event
                * 3 for the 2nd event
                * 4 for the third event
                * 3 for the fourth event
            * Thus, there are 2 * 3 * 4 * 3 = 72 combination meals.
    * Product rule in terms of sets
        * Let A be the set of ways to do the first task and B the set of ways to do 2nd task.
        * If A and B are disjoint, then:
            * The number ways to do both task 1 and 2 can be represented as: $|AxB| = |A| \cdot |B|$
                * The cardinality of the cross product of A and B.
            * In other words: the num elements in the Cartesian product of these sets is the product of number of elements in each set.
* [Addition Rule](permanent/addition-rule.md)
    * Suppose a task 1 can be done n ways and a task 2 can be done in m ways.
    * Assume that both tasks are independent, that is, performing task 1 doesn't mean performing task 2 and vice versa.
    * In this case, the number of ways of executing task 1 or task 2 is equal to n + m.
    * Example
        * The computing department must choose either a student or a member of academic staff as a representative for university committee.
        * How many ways of choose this representative are there if there are 10 academic staff and 77 math students, and no one is both a member of academic staff and a student?
    * Solution
        * By the addition rule, there are 10 + 77 ways of choosing this representative.
* The sum rule in terms of sets
    * Let A be the set of ways to do task 1 and B the set of ways to do task 2, where A and B are disjoint sets.
        * The sum rule can be phrased in terms of sets:
            * $|A \cup B| = |A| + |B|$ as long as $A$ and $B$ are disjoint sets.
* Combining the sum and product rules
    * We can combine the sum and product rules to solve more complex problems.
    * Example:
        * Suppose a letter in a programming language can be either a single letter or a letter followed by 2 digits.
            * What is the number of possible labels?
    * Solution:
        * The number of labels with one letter only is 26
        * Using the product rule the number of labels with a letter folowed by 2 digits is 26 x 10 x10
        * Using the sum rule the total number of labels is 26 + 26>10.10 = 2,626.
* [Subtraction Rule](permanent/subtraction-rule.md)
    * Suppose a task can be done either in one of $n_1$ ways or in one of $n_2$ ways.
    * Then the total number of ways to do the task is $n_1 + n_2$ minus the number of ways common to the two different ways.
    * Also known as the principle of inclusion-exclusion.
        * $|A \cup B| = |A| + |B| - |A \cap B|$
    * Example
        * How many binary bit strings of length eigth either start with a 1 bit or end with the 2 bits 00?
        * Solution:
            * Number of bit strings of length eight that start with a 1 bit: $2.2.2.2.2.2.2 = 2^7 = 128$
            * Number of bit strings of length eight that end with the 2 bits 00: $2^6 = 64$
            * Number of bit trings of length 8 that start with a 1 bit and end with bits 00 is 2^5 = 32
            * Using substraction rule:
                * the number of bit strings either starting with a 1 or ending with 00 is 128 + 64 - 32 = 160.
* [Division Rule](permanent/division-rule.md)
    * Suppose a tak can be done using a procedure that can be carried out in n ways, for every way w, exactly d of the n ways correspond to w.
        * Then this task can be done $n/d$ ways
    * In terms of sets: if the finite set A is the union of n pair-wise disjoint subsets each with d elements, the $n = |A| / d$
    * In terms of functions: if f is a function from A to B, where both are finite sts, and for every value y \in B there are exactly d values x \in A such that $f(x) = y$, then $|B| = |A|/d$
    * Example:
        * In how many ways can we seat 4 people around a table, where 2 seating arrangements are considered the same when each person has the same left and right neighbour
        * Solution:
            * Let's first number the seats around the table from 1 to 4 proceeeding clockwise:
                * There are 4 ways to select the person for seat 1, three for seat 2, two for seat 3 and one for seat 4
                * Thus there are $4.3.2.1 = 24$ ways to order the four people.
                * Since 2 seating arrangments are the same when each person has the same left and right neighbour, for every choice for seat 1, we get the same seating.
                * Therefore, by using the division rule, there are $24/4 = 6$ different seating arrangements.

## Lesson 10.105 The pigeonhole principal

* Pigeonhole principal
    * One of simplest and most useful ideas in maths
    * Let K be a positive integer:
        * If k is a positive integer and k+1 objects are placed into k boxes, then at least one box contains 2 or more objects.
        * Proof by contrapositive:
            * Suppose none of the k boxes has more than 1 object.
            * The total number of objects would be at most k.
            * Which contradicts the statement that we have k+1 objects.
        * Example:
            * If a flock of 10 pigeons roosts in a set of 9 pigeonholes, 1 of the pigeon holes must have more than 1 pigeon.
        * Exercise
            * Prove that a function f from a set with k + 1 elements to a set with k elements is not a one-to-one.
            * Solution: We can prove this using the peigonhole principle:
                * Create a $box_y$, for each element y in the co-domain of f
                * Put all elements x from the domain in the box for y such that f(x) = y
                * Because there are k + 1 elements and only k boxes, at least one box has two or more elements.
                * Therefore, f is not one-to-one.
* The generalised pigeonhole principle
    * If N objects are placed into k boxes, then there is at least one box containing at least ceil(N/k) objects, where ceil(x) is called the ceiling function, which represents the round-up value of x
    * Let's prove it by contrapositive:
        * Suppose none of the boxes contains more than $\lceil N/k \rceil - 1$ objects.
        * Then the total number of objects is at most: $k(\lceil\frac{N}{k}\rceil - 1) < k((\frac{N}{k} + 1) - 1) = N$
        * This is a contradiction because there is a total of N objects.
    * Example:
        * How many cards must be selected from a standard deck of 52 cards to guarantee that at least four cards of the same suit are chosen?
        * Solution:
            * Assume 4 boxes, one for each suit.
            * Using the generalised pigeonhole principle, at least one box contains at least $\lceil \frac{N}{4}\rceil$, where N is the number of cards selected.
            * At least 4 cards of one suit are selected if $\lceil \frac{N}{4} \rceil \geq 4$
            * The smallest integer N such that $\lceil \frac{N}{4} \rceil \geq 4$ is equal to 13.

## Video: 10.107 Permutations and combinations

* Many counting problems can be solved by finding the number of ways to arrange a specified number of distinct elements of a set of particular size, where the order of this element matters, and in some cases doesn't.
* This lecture discusses permutations and combinations, which are used to solve this counting problem.
* [Permutation](../../../../permanent/permutation.md)
    * A permutation of a set of distinct objects is an **ordered arrangement** of these objects.
    * An ordered arrangement of r elements of a set is called an r-permutation.
    * The number of r-permutations of a set with n elements is denoted by $P(n,r)$
    * Example:
        * Let $S = \{1, 2, 3\}$
        * The ordered arrangement 3,1,2 is a 3-permutation of S.
        * The ordered arrangement 3,2 is a 2-permutation of S
        * The 2-permutations of S = {1, 2, 3} are 1,2; 1,3; 2,1; 2,3; 3,1 and 3,2
        * Hence, P(3,2) = 6
    * Number of permutations
        * If n is a positive integer and r is an integer with $r \leq n$, then there are $P(n, r) = n(n - 1)(n - 2) ... (n - (r-1))$ r-permutations of a set with n distinct elements.
        * We can formulate this as:
            * $P(n, r) = \frac{n!}{(n - r)!}$
        * Proof:
            * By product rule:
                * there are n different ways for choosing the 1st element.
                * n-1 ways for choosing the 2nd element.
                * n -3 ways for choosing the 3rd element, etc.
                * there are $(n - (r - 1))$ ways to choose the last element.
                * hence, $P(n, r) = n(n - 1)(n - 2) ... (n - (r - 1))$
                * $P(n, 0) = 1$, since there is only one way to order zero.
            * Example
                * How many possible ways are there of selecting a first prize winner, a 2nd price winner and third-prize winner from 50 different people?
                * Solution:
                    * P(50, 3) = 50 * 49 * 48 = 117,600
* [Combination](../../../../permanent/combination.md)
    * An r-combination of elements of a set is an unordered selection of r elements from the set.
    * An r-combination is a subset of the set with r elements.
    * The number of r-combinations of a set with n distinct elements is denoted by $C(n, r) = \binom{n}{r}$
    * The notation used is called **binomical coefficient**
* Number of combinations
    * The number of r-combinations of a set with n distinct elements can be formulated as:
        * $C(n, r) = \frac{n!}{(n-r)!r!} = \frac{P(n, r)}{r!}$
    * $C(n, r)$ can be referred to as n choose r
    * It follows that $C(n, r) = C(n, n - r)$
    * Example
        * How many ways are there of selecting six players from a 20-member tennis team to make a trip to an intenational competition?
        * $C(20, 6) = \frac{20!}{6!14!} = \frac{20 . 19 . 18 . 17 . 16 . 15}{6 . 5 . 4 . 3 . 2} = 38,760$
