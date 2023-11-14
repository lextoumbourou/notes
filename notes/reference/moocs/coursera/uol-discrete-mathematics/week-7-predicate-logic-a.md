---
title: Week 7 - Predicate Logic A
date: 2022-11-22 00:00
category: reference/moocs
status: draft
parent: uol-discrete-mathematics
modified: 2023-04-09 00:00
---

## 4.101 Introduction to predicate logic

* [Predicate](predicate.md)](../../../../permanent/predicate-logic.md)
    * Propositional logic has some limitations:
        * Cannot precisely express meaning of complex math statements.
        * Only studies propositions, which are statements with known truth values.
    * Predicate logic overcomes the limitations and can be used to build more complex reasoning.
    * Example 1:
        * Given statements:
            * "All men are mortal."
            * "Socrates is a man."
        * Naturally, it follows that: "Socrates is mortal"
        * Propositional logic can't express this reasoning, but predicate logic can enable us to formalise it.
    * Example 2:
        * "x squared is equal to 4."
        * It's not a proposition, as its truth value is a function depending on x.
        * We need predicate logic.

## 4.103 What are Predictates?

* Consider statement $x^2 = 4$
    * It's not a proposition as its true value depends on $x$
    * Therefore, it can't be expressed using propositional logic.
    * Can be expressed using predicate logic.
* [Predicate](../../../../permanent/predicate.md)
    * Predicates behave as functions whose values $T$ or $F$ depend on their variables.
    * Predicates become propositions when their variables are given actual values.
* The statement above has 2 parts:
* The **variable** x: the subject of the statement.
* The **predicate** "squared is equal to 4": the property the subject of the statement can have.
* The statement can be formalised as $P(x)$ where P is the predicate "squared is equal to 4" and x is the variable.
* Evaluate for certain values of x:
    * P(2) is True
    * P(3) is False
* A predictate can depend on multiple values:
    * Let $P(x, y)$ denote "$x^2 > y$":
        * $P(-2, 3)$ = $(4 > 3) \text{ is } \mathbf{ True}$
        * $P(2, 4)$ = $4(2^2 >4) \text{ is } \mathbf{False}$
* Let Q(x, y, z) denote x + y < z
    * Q(2, 4, 5) = (6 < 5) is F
    * Q(2, 4, z) is not a proposition
* Logical operations from propositional logic carry over to predicate logic
    * If $P(x)$ denotes $x^2 < 16$, then:
        * $P(1) \lor P(-5) \equiv (1 < 16) \lor (25 < 16) \equiv T \lor F \equiv T$
        * $P(1) \land P(-5) \equiv T \land F \equiv F$
        * $P(3) \land P(y)$ is not a proposition. It becomes a proposition when y is assigned a value.

## 4.105 Quantification

* [Quantification](permanent/quantification.md)
    * Quantification expresses the extent to which a predicate is true over a range of elements
    * They express the meaning of the words **all** and **some**.
    * Two most important ones:
        * Universal quantifier
        * Existential quantifier
        * Example
            * "All men are mortal"
            * "Some computers are not connected to the network"
    * A third quantifier called "uniqueness quantifier".
* [Universal Quantifier](../../../../permanent/universal-quantifier.md)
    * The universal quantifier of predictate P(x) is proposition:
        * P(x) is true for all values of x in the universe of discourse.
    * We use the notation: $\forall x P(x)$, which is read "for all x"
    * If the universe of discourse is finish, say $\{n_1, n_2, \ldots, n_3\}$ then the universal quanifier is simply the conjunction of the propositions over all elements:
        * $\forall x P(x) \Leftrightarrow P(n_{1}) \land P(n_{2}) \land \ldots \land P(n_k)$
    * Example 1:
        * $P(x)$: "x must take a Discrete Mathematics course"
        * $Q(x)$: "x is a Computer Science student."
        * Where, the university of discourse for both $P(x)$ and $Q(x)$ is all university students.
    * Let's express the following statements:
        * Every CS student must take a discrete math course
            * $\forall \ \  x \ \ Q(x) \rightarrow P(x)$
        * Everybody must take a discrete maths course or be a CS student
            * $\forall \ \ x \ \ (P(x) \lor Q(x))$
        * Everybody must take a discrete maths course and be a CS student
            * $\forall \ \ x \ \ (P(x) \land Q(x))$
    * Example 2:
        * Formalise statement S:
            * S: "For every x and every y, x + y > 10"
        * Let $P(x, y)$ by the statement x + y > 10, where the universe of discourse for x, y is the set of all integers.
        * The statement S is: $\forall x \forall y P(x, y)$
        * Can also be written as: $\forall x, y \ \ P(x, y)$
* [Existential Quantifier](permanent/existential-quantifier.md)
    * The existential quantification of a predicate $P(x)$is the proposition:
        * "There exists a value x in the universe of discourse such that P(x) is true."
    * We use the notation: $\exists \ x \ P(x)$, which reads "there exists x".
    * If the universe of discourse is finite, say $\{n_1, n_2, \ldots, n_k\}$ then the existential quantifier is simply the **disjunction** of propositions over all the elements:
        * $\exists \ x \ P(x) \Leftrightarrow P(n_1) \lor P(n_2) \lor \ldots \lor P(n_k)$
    * Example 1
        * Let $P(x, y)$ denote the statement "x + y = 5".
        * The expression **E**: $\exists \ x \ \exists \ y \ P(x, y)$ means:
            * There exists a value x and a value y in the universe of discourse such that $x + y = 5$ is true.
        * For instance
            * If the universe of discourse is positive integers, E is True.
            * If the universe of discourse is negative integers, E is False.
    * Example 2
        * Let $a, b, c$ denote fixed real numbers.
        * And S be the statement: "There exists a real solution to $ax^2 + bx - c = 0$"
        * S can be expressed as $\exists \ x \ P(x)$ where:
            * $P(x)$ is $ax^2 + bx - c = 0$ and the universe of discourse for x is the set of real numbers.
        * Let's evaluate the truth value of S:
            * When $b^2 >= 4ac, S \text{ is true , as } P(-b \mp \sqrt(b^2 - 4ac)) / 2a = 0$
            * When $b^2 < 4ac, S\text{ is false }$ as there is no real number x that can satisfy the predicate.
* [Uniqueness quantifier](permanent/uniqueness-quantifier.md)
    * Special case of "existential quantifier".
    * The uniqueness quantifier of prediction P of x is the proposition:
        * There exists a unique value of x in the universe such that P of x is true.
        * We use the notation: $\exists ! x \ P(x)$: read as there exists a unique x.
    * Example:
        * Let P(x) denote the statement: $x^2 = 4$
        * The expression $E$: $\exists ! x \ P(x)$ means:
            * There exists a unique value x in the universe of discourse such that $x^2 = 4$ is true.
        * For instance
            * If the universe of discourse is positive integers, E is True (as x = 2 is the unique solution)
            * If the universe of discourse is integers, E is False (as x = 2 and x = -2 are both solutions)

## 4.107 Nested quantifiers

* Nested quantifiers
    * To express statements with multiple variables we use nested quantifiers
        * $\forall x \forall y P(x, y)$ - P(x, y) is true for every pair x, y
        * $\exists x \ \exists y \ P(x,  y)$ - There is a pair x, y for which P(x, y) is true.
        * $\forall x \ \exists y \ P(x, y)$ - For every x, there is a y for whih P(x, y) i true.
        * $\exists x \ \forall y \ P(x, y)$ - there is an x for which P(x, y) is true for every y.
* Binding variables
    * A variable is said to be **bound** if it is within the scope of a quantifier.
    * A variable is **free** if it is not bound by a quantifier or particular values.
    * Example
        * Let P be a propositional function
        * And S the statement: $\exists \ x \ P(x, y)$
        * We can say that:
            * x is bound
            * y is free
* Logical operations
    * Logical operations can be applied to quantified statements
    * Example
        * If P(x) denotes "x > 3" and Q(x) denotes "x squared is even" then
            * $\exists \ x \ (P(x) \lor Q(x)) \equiv T (ex. x = 4)$
            * $\forall \ x \ (P(x) \rightarrow Q(x) \equiv F (ex. x = 5))$
* Order of operations
    * When nested quantifiers are of the same type, the order does not matter.
    * With quantifiers of different types, the order does matter.
    * Example
        * $\forall x \ \forall y P(x, y) \equiv \forall y \ \forall x \ P(x, y)$
        * $\exists x \ \exists y \ P(x, y) \equiv \exists y \ \exists x \ P(x, y)$
        * $\forall x \ \exists y \ P(x, y)$ is different from $\exists y \ \forall x \ P(x,  y)$
* Precendence of quantifiers
    * The quantifiers $\forall$ and $\exists$ have a higher precendence than all logical operators
    * Example
        * P(x) and Q(x) denote two propositional functions.
        * $\forall x \ P(x) \lor Q(x)$ is the disjunction of $\forall x \ P(x) \text{ and } Q(x)$ rather than $\forall x \ (P(x) \text{ and } Q(x))$
