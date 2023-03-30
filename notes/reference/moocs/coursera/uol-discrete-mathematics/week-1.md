---
title: Week 1 - Sets A
date: 2022-10-10 00:00
category: reference/moocs
status: draft
parent: uol-discrete-mathematics
---

# Introduction to Discrete Mathematics

* Dr Lahcen Ouarbya
    * Research area: data analytics
    
* Discrete mathematics
    * Study of discrete objects
        * opposed to connected objects
        * separated and distance from each other
        * integer, positions, set, relationships, functions
    * Increasingly applied to practical fields of math and science
    * Improves reasoning and problem-solving
* This module covers:
    * sets
    * functions
    * logic
    * graphs
    * trees
    * relations
    * combinatorics
    * mathematical induction
    * recursive relations

## Discrete Mathematics syllabus

* Books required:
    * Kenneth, H, Rosen. Discrete Mathematics and its Applications. (2012) Global Edition
    * David Mackinson, Sets, Logic and Maths for Computing, Springer Verlag. 2012 
* Topic 1: Sets
* Topic 2: Functions
* Topic 3: Propositional Logic
* Topic 4: Predicate logic
* Topic 5: Boolean algebra
* Topic 6: Induction & Recursion
* Topic 7: Graphs
* Topic 8: Trees
* Topic 9: Relations
* Top 10: Combinatorics

## 1.104 The definition of a set

* Set theory is a branch of maths dealing with properies of well-defined collections of objects.
* Concept of sets introduced by G. Cantor, a German mathematician.
* Forms the basis of other fields of study
    * counting theory
    * relations
    * graph theory
    * finite state machines
    
* Covered in lesson
    * Definition of set
    * Set notation:
        * Element of a set
        * Cardinality of a set
        * Subset of a set
        
* Definition of set
    * Collection of any kind of objects: people, ideas or numbers, for example.
    * Collection considered "well-defined"
    
* Examples and notation
    * E  = {2, 6, 4, 8}   # Set of positive, even integers < 10.
    * V = {a, e, i, o, u}  # Set of vowels in English alphabet.
    * C = {red, green, blue}  # Set of colours.
    * EmptySet = {}  # Set containing nothing.
    
* Set is an *unordered*, collection of *unique* objects.

* Math notation
    * Use notation $\in$ to represent that something is element of set: $2 \in E$.
    * Use $\notin$ to present not element of set: $3 \notin E$.
 
* Cardinality of set
    * Given set $S$, the cardinality of $S$ is the number of elements contained in $S$.
    * Write cardinality as $|S|$
        * $|C| = 3$
    
* Subset of a set $\subseteq$ 
    * $A$ is a subset of $B$ if and only if every element of $A$ is also in $B$.
        * $A \subseteq B$.
    * This gives us equivalence:
        * $A \subseteq B \iff x \in A \text{ then } x \in B \text{(for all x)}$
 
* Empty set
    * is a subset of any set $\emptyset \subseteq S$
    * empty set is a subset of itself: $\emptyset \subseteq \emptyset$

* Any set is a subset of itself: $S \subseteq S$

* Special sets
    * N = set of natural numbers = {1, 2, 3, 4, ...}
    * Z = set of integers = {..., -3, -2, -1, 0, 1, ...}
    * Q = set of rational numbers (of form a/b where a and b are elements of Z and b $\ne$ 0)
    * R = set of real numbers
    * $N \subseteq Z \subseteq Q \subseteq R$

## 1.106 The listing method and rule of inclusion

* Set representations:
    * Listing method
    * Set builder notation (rules of inclusion)
    
* Listing method
    * Represent a set $S$ using all elements of set $S$.
    * Examples:
        * Set of all vowels in English alphabet.
            * S1 = {a, e, i, o, u}
        * Set of all positive integers less than 10.
            * S2 = {1, 2, 3, 4, 5, 6, 7, 8, 9}
* Set builder notation
    * Example
        * Set of all even integers: { ..., -6, -4, -2, 0, 2, 4, 6 ... }
            * $\text{Even} = \{2n | n \in Z \}$
            * $\text{Odd} = \{2n+1 | n \in Z \}$
        * Set of rational numbers $Q$: { ..., 1/1, 1/2, 1/3, 1/4, ... }
            * $Q = \{ \frac{n}{m} | n,m \in \text{Z and m }\neq 0 \}$
        * Set of things in my bag: {pen, book, laptop, ...}
            * $\text{Bag =} \{x|x \text{ is in my bag } \}$
    * Exercise
        * Rewrite the following sets using listing method
            * $S_1 = \{ 3n | n \in N \text{ and } n < 6\}$
                * $S_1 = {3, 6, 9, 12, 15}$
            * $S_2 = \{2^n|n \in Z \text{ and } 0 \leq n \leq 4 \}$
                * $S_2 = {1, 2, 4, 8, 16}$
            * $S_3 = \{2^{-n}|n \in Z \text{ and } 0 \leq n \leq 4 \}$
                * $S_3 = \{1,\frac{1}{2}, \frac{1}{4}, ... \}$
        * Rewrite the following sets using set building method
            * $S_1 = \{ \frac{1}{2}, \frac{1}{4}, \frac{1}{6}, \frac{1}{8}, ... \}$
                * $S_1 = {\frac{1}{2n} | n \in \text{Z and } 0 < n \leq 5}$

## 1.108 The powerset of a set

* Outline
    * Definition of a powerset of a set.
    * Cardinality of a powerset of a set.
    
* Subset (recap)
    * A set can have another set as its element.
        * $\{5, 6\} \in \{\{5, 6\}, \{7, 8\}\}$
        * $\{5,6\} \subseteq \{5, 6, 7\}$

* Powerset of a set  
    * A set containing all subsets of another set.
    * The powerset of $S$ is $P(S)$ which is the set containing all subsets of $S$.
    * Example
        * $S = {1, 2, 3}$
        * $P(S) = \{ \emptyset, \{1\}, \{2\},\{3\},\{1, 2\},\{1, 3\}, \{2, 3\}, \{1, 2, 3\} \}$
    * Exercise 1
        * $S = \{ a, b \}$
        * $P(S) = \{\emptyset, \{a\}, \{b\}, \{a, b\} \}$
        * $\{a\} \in P(S)$
        * $\emptyset \subseteq P(S)$ - empty set is subset of $P(S)$
        * $\emptyset \in P(S)$ - empty set is also in $P(S)$
    * Exercise 2
        * $P(\emptyset) = ?$
        * $P(P(\emptyset)) = ?$
        * Powerset of empty set is set containing empty set: $P(\emptyset) = \{ \emptyset \}$ 
        * Empty set is the only subset of the empty set: $\emptyset \subseteq \emptyset$
        * Empty set is a set subset of the power set of empty set: $\emptyset \subseteq P(\emptyset)$
        * $P(P(\emptyset)) = \{ \emptyset, \{ \emptyset \} \}$
* Cardinality of a powerset
    * $|P(S)| = 2^{|S|}$
        * $S = \{1, 2, 3\}$, $|S| = 3$, $|P(S)| = 2^3 = 6$
        * $P(S) = \{ \emptyset, \{1\}, \{2\}, \{3\}, \{1, 2\}, \{2, 3\} \}$

## 1.110 Set operations

* Outline:
    * Set operations and membership tables:
        * Set intersection
        * Set union
        * Set difference
        * Symmetric difference
        
* Union operations
    * The union of two sets are all element in *either* A or B.
    *  Notion: $A \bigcup B$
    * Latex operator `\bigcup`
    * The Big Cup; the kind you'd drink at union meetings.
        * Set builder: $A \bigcup B = \{ x | x \in A \text{ or } x \in B$
    * Python example:
    ```python
    A = {1, 2}
    B = {2, 3}
    A.union(B)
    # {1, 2, 3}
    ```
    * Think of a union between people: it's when people come together to make up a larger set.
    * Membership table
        * Show all combinations of sets an element can belong to.
            * Put 1 if element belongs to set.
            * Put 0 if it doesn't
 | $A$   | $B$    | $A \bigcup B$    |
 | --- | ---  | ---    |
 | 0   | 0    | 0       |
 | 0   | 1    | 1       |
 | 1   | 0    | 1       |
 | 1   | 1    | 1       |
 * Intersection
     * Set containing elements in both in set A **and** B.
     * Notion: $A \cap B$
         * Set builder: $A \cap B = \{ x | x \in A \text{ and } x \in B \}$
         * Looks like a horse shoe. The kind you'd take to a dirt road intersection.
     * Think of an intersection between roads: it's the part of road that both of them share.
     * Membership table
 | $A$   | $B$    | $A \bigcup B$    |
 | --- | ---  | ---    |
 | 0   | 0    | 0       |
 | 0   | 1    | 0       |
 | 1   | 0    | 0       |
 | 1   | 1    | 1       |
* Set difference
    * Elements in $A$, but not $B$.
        * $A - B = \{ x | x \in A \text{ and } x \notin B \}$
    * Example:
        * {1, 2} - {2, 3} = {1}
    * Membership table:
 | $A$   | $B$    | $A \bigcup B$    |
 | --- | ---  | ---    |
 | 0   | 0    | 0       |
 | 0   | 1    | 0       |
 | 1   | 0    | 1       |
 | 1   | 1    | 0       |
 
* Symmetric difference
    * Elements in $A$ or in $B$ but not in both.
        * $A \oplus B = \{ x | (x \in A \text{ or } x \in B) \text{ and } x \notin A \cap B \}$
    * Latex: ```\oplus```
    * Can think of it as union of A and B, with all the common elements of A and B removed.
        * $A \oplus B = (A \bigcup B) - (A \cap B)$
    * Example:
        * A = {1, 2, 3} 
        * B = {3, 4, 5}
        * $A \oplus B = \{ 1, 2, 4, 5 \}$
    * Membership table
| $A$ | $B$ | $A \bigcup B$ |
| --- | --- | ------------- |
| 0   | 0   | 0             |
| 0   | 1   | 1             |
| 1   | 0   | 1             |
| 1   | 1   | 0             |

## 1.112 Essential reading

* Koshy, Thomas.  Discrete Mathematics with Applications. (Academic Press, 2004), pp. 67-70 and pp. 72- 75
    * These pages will cover the following topics: the definition of a set; elements of a set; cardinality of a set; listing method; set builder method; subsets of a set; and powersets.
    * Complete the following questions from exercises 2.1
        * pp.76: exercises 1–8, 13-27, 30-32 and 41-44.
    * Link to ebook: https://ebookcentral.proquest.com/lib/londonww/detail.action?docID=294091
