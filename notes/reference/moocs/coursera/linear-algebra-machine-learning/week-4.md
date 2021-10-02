---
title: "Week 4 - Matrices make linear mappings"
date: 2021-09-23 08:30
category: reference/moocs
status: draft
parent: linear-algebra-machine-learning
tags:
  - LinearAlgebra
  - MachineLearningMath
  - GameMath
---

## Matrices make linear mappings

### Matrices as objects that map one vector onto another; all the types of matrices

#### Introduction: Einstein summation convention and the Symmetry of the dot product

* [[Einstein's Summation Convention]] (00:00-04:24)

    * We can represent a [[Matrix Multiplication]] between matrix $A$ and matrix $B$ like this:
    
        $\begin{bmatrix}a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33}\end{bmatrix} \begin{bmatrix}b_{11} & b_{12} & b_{13} \\ b_{21} & b_{22} & b_{23} \\ b_{31} & b_{32} & b_{33}\end{bmatrix} = AB$
        
        We can think of one of the elements in row 2, column 3 of $AB$ as:
        
        ${(ab)}_{23} = a_{21}b_{13} + a_{22}b_{23} + a_{23}b_{33}$
        
        We can rewrite that as the sum of elements $j$:
        
        $ab_{ik} = \sum\limits_{j} a_{ij} \cdot b_{jk}$
        
        In Einstein's convention, if you have a repeated index like $j$ you can remove it alongside the Sigma notation:
        
        $ab_{ik} = a_{ij} \cdot b_{jk}$
        
        The Einstein Summation Convention also describes a way to code [[Matrix Multiplication]].
        
* [[Matrix Multiplication]] with rectangular matrices (04:24-05:55)

    * We can multiply rectangular matrices as long as the columns in the first are equal to the rows in the 2nd.
    
        $A_{(2 \times 3)} \cdot B_{(3 \times 4)}$
        
        The Einstein Summation Convention shows those operations work. As long as both matrices have the same number of $j$s, you can do it.
    
        $C_{ik} = a_{ij}b_{jk}$
    
* [[Dot Product]] revisited using Einstein's Summation Convention (05:55-07:14)
    * The dot product between vectors $\vec{u}$ and $\vec{v}$ in the summation convention is: $u_{i}v_{i}$
        
    * We can also consider a dot product a matrix product of a row vector and a column vector:
    
        ![Dot product as matrix multiplication](/_media/laml-dot-product-as-matrix-mult.png)
        
        So there's some equivalence between [[Matrix Multiplication]] and the [[Dot Product]].
        
* Symmetry of the [[Dot Product]] (07:15-09:32)

    * If we have unit vector $\hat{u} = \begin{bmatrix}u_1 \\ u_2\end{bmatrix}$ and we do a projection onto one of the axis vectors: $\hat{e_1} = \begin{bmatrix}1 \\ 0\end{bmatrix}$, we get a length of $u_1$
    
        ![Projection onto axis vectors](/_media/laml-projection-axis-vector.png)
        
        * If we then project $e_1$ onto $\hat{u}$, we can draw a line of Symmetry between where the two projections cross:

            ![Symmetry of dot product](/_media/laml-symmetry-of-dot-product.png)
            
            * The two triangles on either side are the same size, proving that the projection is the same length in either direction. So that proves the [[Dot Product]] is symmetrical and also that projection is the [[Dot Product]].
                * That explains why matrix multiplication with a vector is considered a projection onto the vectors composing the matrix (the matrix columns).
            
## Matrics transform into the new basis vector set

### Matrics changing basis

* Transforming a [[Vector]] between [[Basis Vectors]] (00:00-08:31)
    * We can think of the columns of a transformation matrix, as the axes of the new basis vectors described in our coordinate system.
        * Then, how do we transform a vector from one set of basis vectors to another?
    * If we have 2 basis vectors that describe the world of Panda bear in  at $\begin{bmatrix}3 \\ 1\end{bmatrix}$ and $\begin{bmatrix}1 \\ 1\end{bmatrix}$. Noting that these vectors are describes in the normal coordinate system with basis vectors $\begin{bmatrix}0 \\ 1\end{bmatrix}$ and $\begin{bmatrix}1 \\ 0\end{bmatrix}$
    
        ![Pandas basis vectors](/_media/laml-pandas-basis-vectors.png)
        
        * In Panda's world, those basis vectors are $\begin{bmatrix}1 \\ 0\end{bmatrix}$ and $\begin{bmatrix}0 \\ 1\end{bmatrix}$
        
    * If we have a vector described in Panda's world as $\frac{1}{2} \begin{bmatrix}3 \\ 1\end{bmatrix}$, we can get it in our frame, by multiplying with Panda's basis vectors:
        $\begin{bmatrix}3 & 1 \\ 1 & 1\end{bmatrix} \begin{bmatrix}\frac{3}{2} \\ \frac{1}{2} \end{bmatrix} = \begin{bmatrix}5 \\ 2 \end{bmatrix}$ 
        * Which begs the question: how can we translate between Panda's world into our world?
        
    * Using the [[Matrix Inverse]] of Panda's basis vector matrix we can get our basis in Bear's world:
    
        $B^{-1} = \frac{1}{2} \begin{bmatrix}1 & -1 \\ -1 & 3\end{bmatrix}$
        
        * If we use that to transform the vector in our world, we should get it in Bear's world.
        
            $\frac{1}{2} \begin{bmatrix}1 & -1 \\ -1 & 3\end{bmatrix} \begin{bmatrix}5 \\ 2\end{bmatrix} = \begin{bmatrix}\frac{3}{2} \\ \frac{1}{2}\end{bmatrix}$
            
* Translate between basis vectors using projections (08:32-11:14)
    * If the new basis vectors are orthogonal then we can translate between bases using only the dot product.

### Doing a transformation in a changed basis

* Doing a transformation of a [[Vector]] in a changed basis (00:00-04:13)
    * How would you do a 45° rotation in Panda's basis?
    * You could first do it in a normal basis.
    
        $\frac{1}{\sqrt{2}} \begin{bmatrix}1 & -1 \\ 1 & 1\end{bmatrix}$
        
    * And multiply that by the vector in our basis:
    
        $\frac{1}{\sqrt{2}} \begin{bmatrix}1 & -1 \\ 1 & 1\end{bmatrix} \begin{bmatrix}3 & 1 \\ 1 & 1\end{bmatrix} \begin{bmatrix}x \\ y\end{bmatrix}$
        
    * That gives us the transformation in our basis vector. You can then multiply that by the inverse of the coordinate matrix, to get it in Panda's coordinate system.
    
        $\frac{1}{2} \begin{bmatrix}1 & -1 \\ -1 & 3\end{bmatrix} \frac{1}{\sqrt{2}} \begin{bmatrix}1 & -1 \\ 1 & 1\end{bmatrix} \begin{bmatrix}3 & 1 \\ 1 & 1\end{bmatrix} \begin{bmatrix}x \\ y\end{bmatrix}$

    * In short: $B^{-1} R B = R_{b}$
