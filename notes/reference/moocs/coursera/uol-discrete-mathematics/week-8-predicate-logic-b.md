---
title: Week 8 - Predicate Logic B
date: 2022-11-29 00:00
category: reference/moocs
status: draft
parent: uol-discrete-mathematics
modified: 2023-04-08 00:00
---

## 4.201 De Morgan's law for quantifiers

* Intuition of [In [Set](permanent/Set.md) theory](permanent/De%20Morgan's%20Law.md)
    * Often we must consider the negation of a quantified expression.
    * Example
        * $S$: "All university's are connected to the network"
        * $P$: "There is at least one computer in the university operating on Linux"
    * Intuitively
        * The negation of S can be verified **if there is at least one** computer not connected to the network.
        * The negation of P can be verified **if all university computers** are not operating on Linux.
    * De Morgan's laws formalise these intuitions.
* [De Morgan's Law for Quantifiers](De Morgan's Law for Quantifiers)
    * The rules for negating quantifiers can be summarised as:
        * $\neg \forall x \ P(x) \equiv \exists x \ \neg P(x)$
        * $\neg \exists x \ P(x) \equiv \forall x \ \neg P(x)$
    * Example:
        * Let S: "Every student of Computer Science has taken a course in Neural Networks"
            * S can be expressed as: $\forall x \ P(x)$
            * U = {students in CS}
            * P(x) = "x has taken course in Neural Networks"
        * The **Negation** of S:
            * It is not the case that every student of CS has taken a course in Neural Networks.:
                * $\neg (\forall x \ P(x)) \equiv \exists x \ \neg P(x)$
            * This implies that: "There is at least one student who has not taken a course in Neural Networks."
    * Example 2:
        * Let R denote: "There is a student in CS who didn't take a course in ML"
            * R can be expressed as: $\exists x \ Q(x)$
            * U = {students in CS}
            * Q(x) = "x didn't take course in ML"
        * The **Negation** of R:
            * It is not the case that there is a student in CS who didn't take a course in ML
            * $\neg (\exists x \ Q(x)) \equiv \forall x \ \neg Q(x)$
            * This implies that: "every student in CS has taken a ML course."
* [Negating Nested Quantifiers](Negating Nested Quantifiers)
    * For nested quantifiers: apply De Morgan's laws from left to right.
    * Example
        * Let $P(x, y, z)$ denote propositional function of variables: x, y and z.
            * $\neg \forall x \  \exists y \ \forall z \ P(x, y, z)$
                * $\equiv \exists x \ \neg \exists y \ \forall z \ P(x, y, z)$
                * $\equiv \exists x \ \forall y \ \neg \forall z \ P(x, y, z)$
                * $\equiv \exists x \ \forall y \ \exists z \ \neg P(x, y, z)$
        * $\neg \forall x \ \exists y \ \forall z \ P(x, y, z)$ is built by moving the negation of the right through all quantifiers and replacing each $\forall$ with $\exists$ and vice versa.

## 4.203 Rules of inference

* [Argument (Logic)](../../../../permanent/Argument (Logic).md)
    * An argument in Propositional Logic is a sequence of [Proposition](../../../../permanent/proposition.md)lled the conclusion
    * The other propositions in the argument are called premises or hypotheses.
* [Valid Argument](../../../../permanent/Valid Argument.md)
    * An argument is valid if the truth of all its premises implies the truth of the conclusion.
    * Example 1
        * "If you have access to the internet, you can order a book on ML"
        * "You have access to the internet"
        * Therefore: "You can order a book on ML"
        * The argument is valid:
            * All premises are true, so the conclusion must be true.

                | Access to internet | Order a book on ML | If you have access to the internet, order a book on ML |
                | ----------------- | ------------------ | ------------------------------------------------------ |
                | 0 | 0 | 1 |
                | 0 | 1 | 1 |
                | 1 | 0 | 0 |
                | 1 | 1 | 1 |

    * The only time that statment is true in row 4, it the time order a book on ML is true.
    * Example 2
        * "If you have acces to the internet, you can order a book on ML"
        * "You can order a book on ML"
        * Therefore: "You have access to the internet"
        * The argument is not valid:
            * there are situations where premises are true and conclusion is false.

            | Access to internet | Order a book on ML | If you have access to the internet, order a book on ML |
            | ----------------- | ------------------ | ------------------------------------------------------ |
            | 0 | 0 | 1 |
            | 0 | 1 | 1 |
            | 1 | 0 | 0 |
            | 1 | 1 | 1 |

* In row 2, the premise is true, but the conclusion is false.
* [Rules of Inference](../../../../permanent/Rules of Inference.md)
    * Building blocks in constructing incrementally complex valid arguments.
    * We can use truth table to figure out if argument is True or False but it's too laborious when you have lots of vars.
        * If you have 8 propositional variables, you would need a truth table with $2^8$ rows.
    * Rules of inference provide simpler way of proving the validity of arguments.
        * Every rule of inference can be proved using a **tautology**.
* [Modus ponens](Modus ponens)
    * Tautology: $(p \land (p \rightarrow q)) \rightarrow q$
    * The rule of inference:
        * $p \rightarrow q$ (if the conditional statement p implies q is true)
        * $p$ (and the conditional statement is true)
        * Then: $q$ (then the conclusion q is also true)
        * Example:
            * p: "It is snowing"
            * q: "I will study Discrete Maths"
            * "If it is snowing, I will study D.M."
            * "It is snowing"
            * Therefore: "I will study Discrete Maths"
* [Modus tollens](Modus tollens)
    * Tautology: $(\neg q \land (p \rightarrow q)) \rightarrow \neg p$
    * The rule of inference:
        * $\neg q$
        * $p \rightarrow q$
        * Then: $\neg p$
        * Example:
            * $p$: It is snowing
            * $q$: I will study Discrete Maths
            * "If it is snowing, I will study Discrete Maths"
            * "I will not study Discrete Maths"
            * Therefore: "It is not snowing"
* [Conjunction](../../../../permanent/conjunction.md)
    * Tautology: $((p) \land (q)) \rightarrow (p \land q)$
    * The rule of inference:
        * $p$
        * $q$
        * $p \land q$
        * Example:
            * $p$: I will study Programming.
            * $q$: I will study Discrete Maths."
            * "I will study Programming."
            * "I will study Discrete Maths"
            * Therefore: "I will study Programming and Discrete Maths"
* [Simplification](Simplification)
    * Tautology: $(p \land q) \rightarrow p$
    * The rule of inference:
        * $p \land q$
        * Therefore: $p$
        * Example:
            * $p$: I will study Discrete Math
            * $q$: I will study Programming.
            * I will study Discrete Math and programming.
            * Therefore: "I will study Discrete Math"
* [Addition](Addition)
    * Tautology: $p \rightarrow (p \lor q)$
    * The rule of inference:
        * $p$
        * Therefore: $p \lor q$
        * Example:
            * $p$: "I will visit Paris"
            * $q$: "I will study Discrete Math"
            * "I will visit Paris"
            * Therefore: "I will visit Paris or I will study Discrete Math"
* [Hypothetical syllogism](Hypothetical syllogism)
    * Tautology: $((p \rightarrow q) \land (q \rightarrow r)) \rightarrow (p \rightarrow r)$
    * The rule of inference:
        * $p \rightarrow q$
        * $q \rightarrow r$
        * Therefore: $p \rightarrow r$
        * Example:
            * $p$: It is snowing
            * $q$: I will study Discrete Maths
            * If it is snowing, I will study Discrete Math.
            * If I study Discrete Math, I will pass the quizzes.
            * Therefore: if it is snowing, I will pass the quizzes.
* [Disjunctive syllogism](Disjunctive syllogism)
    * Tautology: $((p \lor q) \land \neg p) \rightarrow q$
    * The rule of inference:
        * $p \lor q$
        * $\neg p$
        * Therefore: $q$
    * Example:
        * $p$: I will study Discrete Maths
        * $q$: I will study Art.
        * "I will study Discrete Maths or I will study Art"
        * "I will not study Discrete Maths"
        * Therefore: "I will study art"
* [Resolution](Resolution)
    * Tautology: $((p \lor q) \land (\neg p \lor r)) \rightarrow (q \lor r)$
    * The rule of inference:
        * $p \lor q$
        * $\neg p \lor r$
        * Therefore: $q \lor r$
    * Example:
        * $p$: It is raining.
        * $q$: It is snowing.
        * $r$: It is cold.
        * "It is raining or it is snowing."
        * "It is not raining or it is cold."
        * Therefore: "It is snowing or it is cold."
* [Building Valid Arguments](Building Valid Arguments)
    * To build a valid argument, we need to follow these steps:
        * If initially written as English, transform into argument form by choosing a variable for each simple proposition.
        * Start with the hypothesis of the argument
        * Build a sequence of steps in which each step follows from the previous step by applying:
            * rules of inference.
            * laws of logic.
        * The final step of the argument is the conclusion.
    * Example 1
        * Building a valid argument from these premises:
            * "It is not cold tonight"
            * "We will go to the theatre only if it is cold."
            * "If we do not go to the theatre, we will watch a movie at home."
            * "If we watch a movie at home, we will need to make popcorn."
        * Prop variables:
            * p: It is cold tonight.
            * q: We will go to the theatre.
            * r: We will watch a movie at home.
            * s: We will need to make popcorn.
        * $\neg p$: It is not cold tonight.
        * $q \rightarrow p$: We will go to the theatre only if it is cold.
        * $\neg q \rightarrow r$: If we do not go to the theatre, we will watch a movie at home.
        * $r \rightarrow s$: If we watch a movie at home, we will need to make popcorn.
    * Example 2
        * 1. $q \rightarrow p$ - Hypothesis
        * 2. $\neg p$ - Hypothesis
        * 3. $\therefore$ $\neg q$ - Modus tollens 1, 2
        * 4. $\neg q \rightarrow r$ - Hypothesis
        * 5. $\therefore$ $r$ - Modus ponens 3,4
        * 6. $r \rightarrow s$ - Hypothesis.
        * 7. $\therefore s$ - Modus ponens 5, 6
        * Conclusion: we will need to make popcorn.
* [Logical Fallacies](Logical Fallacies)
    * A fallacy is the use of incorrect argument when reasoning
    * Formal fallacies can be expressed in propositional logic and proved to be incorrect.
    * Some of the widely use formal fallacies are:
        * affirming the consequent
        * a conclusion that denies premises
        * contradictory premises
        * denying the antecedent
        * existential fallacy
        * exclusive premises
    * Example
        * Consider the argument:
            * If you have internet acces, you can order this book.
            * You can order this book.
            * Therefore, you have internet access.
        * This argument can be formalised as: if $p \rightarrow q$ and $q$ then $p$
        * Where
            * $p$: You have internet access
            * $q$: You can order this book
            * The proposition $((p \rightarrow q) \land q) \rightarrow p$ is not a tautology, because it is false when p is false and q is true.
            * This is an incorrect argument using the fallacy of affirming the consequent (or conclusion).

## 4.205 Rules of inference with quantifiers

* [Rules of Inference with Quantifiers](Rules of Inference with Quantifiers)
    * Previously introduced rules of inference for propositions.
    * Now describe important rules of inference for statements involving quantifiers.
    * These rules of inference remove or reintroduce quantifiers within a statement.
* [Universal Instantiation (UI)](Universal Instantiation (UI))
    * The rule of inference:
        * $\forall P(x)$
        * $\therefore P(c)$
    * Example:
        * All comp science students study discrete maths.
        * $\therefore$ Therefore, John, who is a computer science student, studies discrete math.
* [Universal Generalization (UG)](Universal Generalization (UG))
    * The rule of inference:
        * $P(c)$ for an arbitrary element of the domain.
        * $\forall x P(x)$
        * Example:
            * DS = {all data science students}
            * Let c be an arbitrary element in DS.
            * c studies ML.
            * $\therefore$ Therefore $\forall x \in  \text{DS}$, $x$ studies ML.
* [Existential Instantiation (EI)](Existential Instantiation (EI))
    * The rule of inference:
        * $\exists x \ P(x)$
        * $\therefore P(c)$ for some element of the domain.
    * Example:
        * DS = {all data science students}
        * There exists a student of data science who uses Python Pandas Library.
        * Therefore, there is a student $c$ who is using Pandas.
* [Existential Generalization (EG)](Existential Generalization (EG))
    * The rule of inference:
            * $P(c)$ for some element of the domain.
            * Therefore, $\exists x P(x)$
    * Example:
        * DS = {all data science students}
        * John, a student of data science, got a A in ML.
        * Therefore, there exists someone in DS who got an A in ML.
* [Universal Modus Ponens](Universal Modus Ponens)
    * The rule of inference:
        * $\forall x P(x) \rightarrow Q(x)$
        * $P(a)$ for some element of the domain.
        * $Q(a)$
    * Example:
        * DS = {all comp sci students}
        * Every computer science student studying data science will study ML.
        * John is a computer science student studying data sciecnce.
        * Therefore, John will study ML.
* [Universal Modus Tollens](Universal Modus Tollens)
    * The rule of inference:
        * $\forall x P(x) \rightarrow Q(x)$
        * $\neg Q(a)$ for some element of the domain.
        * $\neg P(a)$
    * Example
        * CS = {all computer science students}
        * Every computer science student studying data science will study machine learning.
        * John is not studying machine learning.
        * Therefore, John is not studying data science.
* Expressing complex statements
    * Given a statement in natural language, we can formalise it using the following steps as appropriate:
        * 1. Determine the universe of discourse of variables.
        * 2. Reformulate the statements by making "for all" and "there exists" explicit
        * 3. Reformulate the satement by introducing variavbles and defining predicates.
        * 4. Reformulate the statement by introducing quantifiers and logical operations.
    * Example 1
        * Express the statement S: "There exists a real number between any two not equal real numbers".
        * The universe of discourse is: real numbers.
        * Introduce variables and predicates:
            * "For all real numbers x and y, there exists z between x and y."
        * Introduce quantifiers and logical operations:
            * $\forall x \ \forall y$ if $x < y$ then $\exists z$ where $x < z < y$
    * Example 2
        * Express the statement S: "every student has taken a course in machine learning".
        * The expression will depend on the choice of the universe of discourse.
        * Case 1: U = {all students}
            * Let M(x) be: "x has taken a course in ML."
            * S can be expressed as: $\forall x \ M(x)$
        * Case 2: U = {all people}
            * Let S(x) be: "x is a student" and M(x) the same as in case 1
            * S can be expressed as $\forall x (S(x) \rightarrow M(x))$
            * Note: $\forall x (S(x) \land M(x))$ is not correct.
    * Example 3
        * Express the statement S: "some student has taken a course in machine learning".
            * The expression will depend on the choice of the universe of discourse.
        * Case 1: U = {all students}
            * Let M(x) be: "x has taken a course in ML."
            * S can be expressed as: $\exists x M(x)$
        * Case 2: U = {all people}
            * Let S(x) be: "x is a student" and M(x) the same as in case 1.
            * S can be expressed as $\exists x (S(x) \land M(x))$
            * Note: $\exists x (S(x) \rightarrow M(x))$ is not correct.

## Problem sheets

**Question 1.**

Let $P(x)$ be the predicate "$x^2 > x$" with the domain the set $R$ of all real numbers. Write $P(2)$, $P(\frac{1}{2})$, and $P(-\frac{1}{2})$ and indicate which are true and false.

$P(2) = 4 > 2 = T$
$P(\frac{1}{2}) = \frac{1}{2}^2 > \frac{1}{2} = \frac{1}{4} > \frac{1}{2} = F$
$P(-\frac{1}{2}) = \frac{1}{4} > -\frac{1}{2} = T$

**Question 2.**

Let $P(x)$ be the predicate "$x^2 > x$" with the domain the set of $\mathbb{R}$ of all real numbers.

What are the values $P(2) \land P(\frac{1}{2})$ and $P(2) \lor P(\frac{1}{2})$?

$P(2) \land P(\frac{1}{2}) = (4 > 2) \land (\frac{1}{4}  > \frac{1}{2}) = F$
$P(2) \lor P(\frac{1}{2}) = (4 > 2) \lor (\frac{1}{4}  > \frac{1}{2}) = T$

**Question 3.**

1. Let D = {1, 2, 3, 4} and consider the following statement: $\forall x \in D, x^2 \ge x$. Write one way to read this statement and show that it is true.

* $1^2 = 1$, $1 \ge 1$
* $2^2 = 4$, $4 \ge 2$
* $3^2 = 6$, $6 \ge 3$
* $4^2 = 8$, $8 \ge 4$

The statement is true for all $D$, hence it is true.

2. $\forall x \in \mathbb{R}, x^2 \ge x$

$x = \frac{1}{2}$ $\frac{1}{2}^2 = \frac{1}{4}$, $\frac{1}{4} >= \frac{1}{2} = F$

**Question 4.**

1. Consider the following statement:

    $\exists n \in \mathbb{Z}^{+}$ such that $n^2 = n$

    Write one way to read this statement, and show that is it true.

    $n = 1$, $n^2 = 1$, $1 = 1$ therefore, the statement is true.

2. Let $E = \{5, 6, 7, 8\}$ and consider the following statement: $\exists n \in E, n^2 = n$

   $5^2 = 25$, $25 \ne 5$
   $6^2 = 36$, $36 \ne 6$
   $7^2 = 49$, $49 \ne 7$
   $8^2 = 64$, $64 \ne 8$

   Therefore, the statement $\exists n \in E, n^2 = n$ is false.

**Question 5.**

Rewrite each of the statements formally, using quantifiers and variables.

1. All triangles have three sides.

$\forall t \in \mathbf{T}$, $t$ has 3 sides (where $\mathbf{T}$ is the set of all triangles).

2. No dogs have wings.

$\forall d \in \mathbf{D}$, $d$ doesn't have wings (where $\mathbf{D}$ is the set of all dogs).

3. Some programs are structured.

$\exists{p} \in \mathbf{P}$, $p$ is a structured program (where $\mathbf{P}$ is the set of all programs).

**Question 6.**

Rewrite the following statements in form of $\forall$ ___ if ___ then ___

1. If a real number is an integer, then it is a rational number.

$\forall$ real number $x$, if $x$ is an integer, then $x$ is a rational number.

2. All bytes have eight bits.

$\forall x$, if $x$ is a byte, then $x$ has eight bits.

3. No fire trucks are green.

$\forall x$, if $x$ is a fire truck, then $x$ is not green.

**Question 7.**

A prime number is an integer greater than 1 whose only positive integer factors are itself and 1.

Consider the following predicate Prime(n): n is prime and Even(n): n is even.

Use the notation Prime(n) and Even(n) to rewrite the following statement: "There is an integer that is both prime and even"

$\exists n$ such that $\mathbf{Prime}(n) \land \mathbf{Even}(n)$

**Question 8.**

Determine the truth value of each of the following where $P(x, y)$ : $y < x^2$, where $x$ and $y$ are real numbers:

1. $(\forall x) (\forall y) P(x, y)$

This is false as there exists, $x, y \in \mathbb{R}$ where $x = 1$ and $y = 1$, such that P(1, 1) is false.

2. $(\exists x)(\exists y) P(x, y)$

True. There exists $x, y \in \mathbf{R}$ where $x = 4$ and $y = 2$ such that P(x, y) is true.

3. $(\forall y)(\exists x) P(x, y)$

True. For all $y \in \mathbb{R}$ there exists $x = 2 |\sqrt{|y|}|$ with $x^2 = 4|y| > y$

4. $(\exists x)(\forall y) P(x, y)$

False. There is no $x$ that doesn't have a $y$ that is smaller than $x^2$.

**Question 9.**

Let P(x) denote the statement x is taking discrete math.

The domain of discourse is the set of all students.

Write the statements in words:

$\forall x P(x)$ - every student is taking the Discrete Maths course.
$\forall x \ \neg P(x)$ - every student is not taking a Discrete Maths course.
$\neg (\forall x P(x))$ - it is not the case that every student is taking the Discrete Maths course.
$\exists x P(x)$ - there exists one student who is taking a Discrete Maths course.
$\exists x \ \neg P(x)$ - some student is not taking a Discrete Maths course.
$\neg(\exists x \ P(x))$ - no student is taking a Discrete Maths course.

**Question 10.**

Let P(x) denote the statement 'x is a professional athelete' and let Q(x) denote the statement "x plays football".

The domain of discourse is the set of all people.

Write the following in words:

1. $\forall x (P(x) \rightarrow Q(x))$

Every professional athlete plays football.

2. $\exists x (Q(x) \rightarrow P(x))$

Either someone does not play football or some football player is a pro athlete.

3. $\forall x (P(x) \land Q(x))$

Every one is a professional athlete and plays football.

**Question 11.**

Let $P(x)$ denote the statment "x is a professional athlete" and let $Q(x)$ denote the statement "x plays football".

The domain of dis-course is teh set of all people.

Write the negation of each proposition symbolically and in words.

1. $\forall x (P(x) \rightarrow Q(x))$

$\exists x \ \neg(P(x) \rightarrow Q(x))$

There are people who are professional athletes that don't play football.

2. $\exists x (Q(x) \rightarrow P(x))$

$\forall x \ \neg(Q(x) \rightarrow P(x)))$

It is not the case that if you play football you are a professional athere.

3. $\forall x (P(x) \land Q(x))$

$\exists x (\neg P(x) \lor \neg Q(x)))$

There are people who are either not profressional athleses or not football players.

**Question 12.**

Let $P$ and $Q$ denote the propositional functions:

P(x): x is greater than 2.
Q(x): x^2 is greater than 4.

where, the universe of discourse for both P(x) and Q(x) is the set of real number, $\mathbb{R}$

1. Use quantifiers and logical operators to write the following statement formally:

"if a real number is greater than 2, then its square is greater than 4"

$\forall x (P(x) \rightarrow Q(x))$

2. Write a formal and informal contrapositive, converse and inverse of the statement above in (1).

Contrapositive: $\forall x \ (\neg Q(x) \rightarrow \neg P(x))$

If the square of x is not greater than 4, then x is not greater than 2.

Converse: $\forall x \ (Q(x) \rightarrow P(x))$

If the square of x is greater than 4, than x is greater than 2.

Inverse: $\forall x (\neg P(x) \rightarrow \neg Q(x))$

If x is less than or equal to 2, then the square of x is less than or equal to 4.

**Question 13.**

Rewrite each of the following statements in English as simply as possible without using the symbols $\forall$ or $\exists$ variables.

1. $\forall$ color $c$, $\exists$ an animal $a$ such that $a$ is colored $c$.

For every colour, we can find an animal that has that colour.

2. $\exists$ a book $b$ such that $\forall$ person $p$, $p$ has read $b$.

There is at least one book that every person has read.

3. $\forall$ odd integer $n$, $\exists$ an integer $k$ such that $n = 2k + 1$

For every odd integer $n$, there is an integer k where $n = 2k + 1$

4. $\forall x \in \mathbb{R}$, $\exists$ a real number $y$ such that $x + y = 0$

For every read number x, there is another number y where adding them together gives 0.

**Question 14.**

Rewrite the statement "No good cars are cheap" in the form $\forall x$ if $P(x)$ then $\neg Q(x)$

$\forall x$ if $x$ is a good car, then $x$ is NOT cheap.

Indicate whether each of the following arguments is valid or invalid and justify your answers.

1. No good cars are cheap
A Ferrari is a good car.
$\therefore$ A Ferrari is not cheap

This is a valid argument using [Universal Modus Ponens](Universal Modus Ponens) or [Universal Instantiation (UI)](Universal Instantiation (UI))

2. No good cars are cheap.
A BMW is not cheap.
$\therefore$ a BMW is not a good car.

This is invalid. Converse error.

**Question 15.**

Let x be any student and C(x), B(x) and P(x) be the following statements:

C(x): “x is in this class“.
B(x): “x has read the book”.
P(x): “x has passed the first exam”.

Rewrite the following symbolically and state whether it a valid argument.

A student in this class has not read the book

$\exists x (C(x) \land \neg B(x))$

Everyone in this class passed the first exam

$\forall x (C(x) \rightarrow P(x))$

∴ Someone who passed the first exam has not read the book

$\therefore \exists x (C(x) \rightarrow \neg B(x))$

This is valid (not sure why).

# 4.209 Predicate logic - Peer-graded

## Question

 Use the following rules of inference: existential instantiation, universal instantiation, disjunctive syllogism, modus tollens and existential generalisation to show that:

**if**:

Hypothesis 1: $\forall x (P (x) \lor Q(x))$
Hypothesis 2: $\forall x (¬Q(x) ∨ S(x))$
Hypothesis 3: $\forall x(R(x) \rightarrow \neg S(x))$
Hypothesis 4 : $\exists x \neg P(x)$ are true

**then**:

Conclusion: $\exists x¬R(x)∃x¬R(x)$ is true.

## Answer

We can use the following steps:

1. Existential instantiation: Since $\exists x \ \neg P(x)$ is true, there exists some specific object $a$ such that $\neg P(a)$.
2. Universal instantiation: Since $\forall x (P(x) \lor Q(x))$ is true, we can instantiate this to $P(a) \lor Q(a)$
3. Disjunctive syllogism: Since we know that $\neg P(a)$ from step 1, and $P(a) \lor Q(a)$ from step 2, we can conclude that $Q(a)$ must be true.
4. Universal instantiation: Since $\forall x \ (\neg Q(x) \lor S(x))$, we can instantiate this to $\neg Q(a) \lor S(a)$.
5. Disjunctive syllogism: since we know that $Q(a)$ is true from step 3, and $\neg Q(a) \lor S(a)$ is true, we can conclude that $S(a)$ must be true.
6. Universal instantiation: since $\forall x (R(x) \rightarrow \neg S(x)))$ is true, we can instantiate this to $R(a) \rightarrow \neg S(a)$
7. Modus tollens: since we know that $S(a)$ is true from step 5, and $R(a) \rightarrow \neg S(a)$ from step 6, we can conclude that $\neg R(a)$ must be true.
8. Existential generalisation: Since $\neg R(a)$ is true for a specific object $a$, we can conclude that $\exists x \neg R(x)$ is true.
