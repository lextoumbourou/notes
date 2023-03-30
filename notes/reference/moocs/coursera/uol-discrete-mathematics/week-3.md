---
title: Week 3 - Topic 02 A. Functions
date: 2022-10-23 00:00
category: reference/moocs
status: draft
parent: uol-discrete-mathematics
---

## 2.101 Introduction

* In everyday life, many quantities depend on change in variables:
    * Plant's growth depends on sunlight and rainfall.
    * A runner's speed == how long it takes to run a distance.
* A function is a rule that related how one quantity depends on another.
    * It's a central concept in programming.
 
## 2.102 The definition of a function

* A function is a relationship between a set of inputs and a set of outputs, so inputs map to exactly one output.
    * Important concept in Computer Science.
* Outline
    * Definition of a function
    * Domain, co-domain and range of a function.
    * Image and pre-image (antecendent) of an element.
* Definition of a function
    * Function is a "well-behaved relation"
        * Given a starting point, we know exactly where to go.
    * A function $f$ from a set of $A$ to a set of $B$ is an assignment of exactly one element of $B$ to each element of $A$.
        * If $f$ is the function from A to B, we write: $f: A \rightarrow B$
        * We can read this as f maps A to B.
            * $x \in A: \ \ x \rightarrow f(x) = y \ \ \ (y \in B)$
* Terminology
    * Given the function above, $A$ is the set of all inputs and called the "domain" of $f$.
        * Written as $D_f = A$ 
    * $B$ is the set containing the outputs and called the co-domain of $f$.
        * Written as $\text{co-}D_f = B$ 
    * The set of all outputs is called the range of f and is written as $R_f$.
    * $y$ is called the image of $x$.
    * $x$ is called the pre-image of $y$.
    
    ![[domain-codomain-range.png]]
* Example: a set mapping characters to a length.
    * $f(\text{Sea}) \rightarrow 3$ (contains 3 characters)
    * $f(Land) \rightarrow 4$ (contains 4 characters)
    * $f(on) \rightarrow 2$
        * 2 is the image of "on"
        * "on" is the pre-image of 2.
* Conditions under which a relation is not a function:
    * Some inputs do not have an image.
    * Some inputs have more than one image.
* Exercise 1
    * Given the following function: $f: Z \rightarrow Z$ with $f(x) = |x|$, what is domain, co-domain and range for function $f$?
        * Domain: $Z$
        * Co-domain: $Z$
        * Range: $Z^{+}$
* Exercise 2
    * Given the following function: $g: R \rightarrow R$ with $g(x) = x^2 + 1$
        * Domain: R
        * Co-domain: R
        * Range: {1, 5, 9 ...}
        * Pre-images(5) = {-2, 2}

## 2.104 Plotting functions

* Outline:
    * Linear functions
    * Quadratic functions
    * Exponential functions
* Linear function is of form: $f(x) = ax + b$
    * Where $a$ and $b$ are real numbers.
    * Straight-line function that passes through point (0, b).
    * $a$ is the gradient of the function. Where $a > 0$ the function is increasing.
        * That is: $x_1 \leq x_2$ then $f(x_1) \leq f(x_2)$.
    * Example of increasing linear function
      ![[linear-increasing.png]]
      * When the gradient is < 0, the function is decreasing 
* $f: R \rightarrow R$
    * $f(x) = ax +b$
        * If $a > 0$ then function is increasing.
        * If $x_1 \leq x_2$ then $f(x_1) \leq f(x_2)$
        * ![[linear-decreasing.png]]
* Quadratic functions: $f(x) = ax^2 + bx + c$
    *  Where $a$, $b$ and $c$ are real numbers and $a \ne 0$.
        ![[quadratic.png]]
    * Domain of function f(x) is set of real numbers.
    * Range of function is set of positive numbers.
* Laws of [[Exponential Functions]].
    * $b^xb^y = b^{x + y}$
    * $\frac{b^x}{b^y} = b^{x-y}$
    * $(b^x)^y = b^{xy}$
    * $(ab)^x = a^xb^x$
    * $(\frac{a}{b})^x = \frac{a^x}{b^x}$
    * $b^{-x} = \frac{1}{b^x}$
* Graph of [[Exponential Function]].
    * If base $b$ in $f(x) = b^x$, $b > 1$ then function is increasing and represents growth shown in this graph:
          ![[exponential-growth-function.png]]
        * Graph also shows that the point $(0,1)$ is a "common point".
        * Domain is equal to set of all real numbers.
        * Range is equal to set of all real positive numbers.
        * X-axis is horizontal asymtot to curve of function.
    * If base 0 < b < 1, then function is decreasing
          ![[exponential-decay-function.png]]
          * Domain and range are the same as previous function.

## 2.106 Injective and surjective functions

* Outlines
    * Injective or "one-to-one" functions.
    * Surjective or "onto" functions.
* [[Injective Function]]
    * f is injective (one-to-one) function if and only if:
        * any 2 distinct inputs will lead to 2 distinct outputs.
        * In other words:
            * for all $a, b \in A, \text{ if } a \ne b \text{ then } f(a) \ne f(b)$
            * same as saying: $a, b \in A, \text{ if } f(a) = f(b) \text{ then } a = b$
            
    
    *  *![[injective-function.png]]
        * Example on the left is an injective function, as every element of $A$ has a unique image in B.
        * Example on the right is not injective. 2 or 4 in A have the same image 0. 1 and 3 have the same image 1.
    * You can show a function is not injective by finding two different inputs $a$ and $b$ with the same [[Function Image]].
    * An example from a [[Linear Function]].
        * Show function $f: R -> R$ with $f(x) = 2x + 3$ is an [[Injective Function]].
        * Proof 1:
            * Let $a, b \in R$, show that $\text{ if } f(a) = f(b) \text{ then } a = b$
            * $f(a) = f(b)$ => $2a + 3 = 2b + 3$ => $2a = 2b$ => $a = b$ => f is injective.
        * Proof 2:
            * Let $a, b \in R$, show that $\text{ if } a \ne b \text{ then } f(a) \ne f(b)$
                * $a \ne b$ => $2a \ne 2b$ => $2a + 3 \ne 2b+3$ => $f(a) \ne f(b)$ => f is injective
    * An example quadratic function that is not injection.
        * Show function $f: R -> R$ with $f(x) = x^2$ is not an [[Injective Function]] (not one-to-one).
            * We only need to find 1 counter example with 2 counter examples that have the same image.
            * One example is 5 and -5 have the same image.
                * $f(5) = (5)^2 = (-5)^2 = f(-5)$
                    * Since: $-5 \ne 5$ it's not injective.
                    * If we change domain to $R^{+}$, the function becomes injective.
            * Proof 1:
                * Let $a, b \in R^{+}$ show that if $f(a) = f(b)$ then $a = b$.
                    * Let $a, b \in R^{+}$ show that if $f(a) = f(b)$ then $a = b$
            * Proof 2:
                * Let $a, b \in R^{+}$ show that if $a \ne b$ then $f(a) \ne f(b)$
                * a \ne b => a^2 \ne b^2 as a, b \in R+ => f(a) \ne f(b) => f is injective.
* [[Surjective Function]]
    * A function is said to be a *surjective* (onto) function if and only if every element of the co-domain of $f$, $B$, has at least one pre-image in the domain of $f, A$.
        * In other words, every element in the output domain has some input that will return it.
    * for all $y \in B$ there exists $x \in A$ such that $y = f(x)$
        * Equivalent to saying range and co-domain of surjective function are the same.
            * $\text{ CO}-D_f = R_f$
        * Examples:
            ![[surjective-example.png]]
    * An example [[Linear Function]]
        * Show that the function $f: R -> R$ with $f(x) = 2x+3$ is a surjective (onto) function.
        * Need to show that for any element $y \in R$, there exists $x \in \mathbb{R}$ such that $f(x) = y$
    * Proof:
        * $f(x) = y$ => $2x + 3 = y$ => $2x = y - 3$ => $x = \frac{y-3}{2} \in R$
        * Hence, for all $y \in R$, there exists $x = \frac{y-3}{2} \in R$ such that $f(x) = y$
    * An example quadratic function that is not surjective
        * Show that function $f: R-> R$ with $f(x) = x^2$ not a surjective (onto) functions
        * Proof:
                * Let $y \in R$, show that there exists $x \in R$ such that $f(x) = y$
                * $R_f (\text{ set images }) = [0, + \infty [\ne R(co-D_f) = R$
                    * We know the range of $Rf$ is positive integers only: all negative images have no pre-images. 
* Examples
    * Injective, not surjective
          ![[injective-not-surjective.png]]
        * Injective because each element in the domain has a unique image.
        * Not surjective because the element 2 in the co-domain has no pre-image.
    * Surjective but not injective
        ![[surjective-not-injective.png]]
        * Not injective because a and d are different but have the same image.
    * Injective and surjective
        ![[injective-and-surjective.png]]
        * Each element has a unique image.
        * Each element in co-domain has at least one pre-image.
    * Neither injective nor surjective
    ![[not-injective-or-surjective.png]]
        * Not injective because a and c have the same image.
        * Not surjective because the 4 element of co-domain has no pre-image.
    * Not a valid function
      ![[not-valid-function.png]]
      * Input a has 2 outputs. In a function, an input can only have a single output.
