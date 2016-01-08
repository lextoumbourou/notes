# Linear Algebra

## Matrices and Vectors

* Denoted by ``n x m`` (n rows, m columns).

## Special Matrices

* Square matrix: a matrix whose dimensions are ``n x n``.
* Zero matrix: denoted by ``0n x m``, a matrix where all the values are 0. (acts like 0 in scalar land)
* Identity matrix: main diagonals are 1 and others 0. When multipling with another matrix, the result is the other matrix. (acts like 1 in scalar land)
* Column matrix: ``n x 1`` and row matrix: ``1 x n``. Aka vector.

## Arithmetic

* Adding or subtract matrices: add or subtract each field. Must be the same size.
* Scalar multiplication: multiply each value in matrix by scalar.
* Matrix multiplication: ``A x B`` A must have same number of columns as B does rows eg ``A(2 x 3) * B(3 x 2)`` is valid. The resulting size will be ``2 x 2``.
  * Example:

     ```
      A = [1, 2, 4]   B = [4, 2]
          [4, 3, 5]       [5, 8]
                          [7, 1]

           (2 x 3)        (3 x 2)
   
      Outcome = [(1 * 4) + (2 * 5) + (4 * 7)]
                [(1 * 2) + (2 * 8) + (4 * 1)]
                [(4 * 4) + (3 * 5) + (5 * 8)]
                [(4 * 2) + (3 * 8) + (5 * 1)]
      ```
  * Commutative property does not apply in matrix multiplication: order matters.

## Determinant

* Function that takes a square matric and convert it to a number. Formula loks like:

```
[a c]
[b d] == a*d - c*b
```

## Matrix Inverse

* Given a square matrix ``A`` of size ``n x n`` we want to find another matrix such that:

```AB = BA = I n``

## Matrix calculus

* Each of the points in a matrix can be function, can determine the derivative of each point.
