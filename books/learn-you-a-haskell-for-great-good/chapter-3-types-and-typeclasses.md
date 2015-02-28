# Chapter 3: Types and Typeclasses

## Believe the Type

* Haskell is static typed
  * Has type inference
* Use ```:t``` command to get variable's type.
```
> :t "a"
'a' :: Char
> :t True
True :: Bool
relude> :t (True, "blah")
(True, "blah") :: (Bool, [Char])
```
* Functions can have types, either inferered or declared (best practice):
```
removeNonUppercase :: [Char] -> [Char]
removeNonUppercase st = [ c | c <- st, elem c ['A'..'Z']]
```
  * ```removeNonUppercase``` accepts a string and outputs another string. 
* Common types:
  * ``Int`` - integer. Bounded by system size.
  * ``Integer`` - alias for ``Int``.
  * ``Float`` - floating point type with single precision.
  * ``Double`` - floating point with double precision.
  * ``Bool`` - boolean type
  * ``Char`` - represents a single character. List of chars is a string.
  * ``Tuple`` - standard tuple type.
* Types always begin with an uppercase character. 
```
*Main> :t 3.14
3.14 :: Fractional a => a
*Main> :t 'B'
'B' :: Char
```

## Generic types

* When a type doesn't begin with a uppercase, it can be assigned any type, aka, generic.

## Typeclasses 101

* Typeclass is "sort of interface that defines some behaviour".
* Equality operator ``==`` is a function.
```
*Main> :t (==)
(==) :: Eq a => a -> a -> Bool
```
  * Everything before ```=>``` is a "class constraint".
  * ```(Eq a)``` - Type of variable must be memory of ```Eq``` class.
    * All standard Haskell types are part of ```Eq``` typeclass (except IO handlers).
  * ```a -> a -> Bool``` - take two vals and return a bool.
  * Note: If a function is comprised of only special chars, it's an infix function by default. 
* Basic typeclasses:
  * ```Eq```
    * For stuff that supports equality testing.
    * Functions members implement are ``==`` and ``/=``.
  * ```Ord```
    * For types that have an ordering (for comparison functions).
    * To be ```Ord``` it must also be a member of ```Eq```.
    * ``compare`` function takes two ``Ord`` members and returns ``Ordering``
      * ``Ordering`` is a type that can be ``GT`` (greater than), ``LT`` (lesser than) or ``EQ`` (equal):
        ```
        Prelude> "Blah" `compare` "Moo"
        LT
        Prelude> 5 `compare` 3
        GT
        Prelude> 5 `compare` 6
        LT
        ```
  * ``Show``
    * Members can be presented as strings. 
    * Most used function that deals with ``Show`` typeclass is ``show``
      ```
      Prelude> show "Hello, bro."
      "\"Hello, bro.\""
      Prelude> show 100
      "100"
      Prelude> show False
      "False"
      ```
  * ``Read``
    * Opposite of ``Show``.
    * ``read`` function takes a string and returns a member of ``Read``

      ```
      Prelude> read "True" || False
      True
      Prelude> read "8.2" + 3.8
      12.0
      Prelude> read "[10,12,14]" ++ [16]
      [10,12,14,16]
      ```
    * Trying to read a single value will fail, because Haskell can't infer what you'd like to do with that value:

    ```
    Prelude> read "4"
    *** Exception: Prelude.read: no parse
    ```

    * The type signature of read, returns a type that's part of ``Read``:

    ```
    Prelude> :t read
    read :: Read a => String -> a
    ```

    * Can use **type annotations** to force a type:

    ```
    Prelude> read "5" :: Float
    5.0
    Prelude> read "5" :: Int
    5
    ```
  * ``Enum``
    * Ordered sequential types.
    * Have successors (get with ``succ``) and predecesors (get with ``pred``)
    ```
    Prelude> succ 'Z'
    '['
    Prelude> succ 'A'
    'B'
    Prelude> pred 'B'
    'A'
    ```
  * ``Bounded``
    * Have upper and lower bound:

    ```
    Prelude> minBound :: Int
    -9223372036854775808
    Prelude> maxBound :: Char
    '\1114111'
    Prelude> maxBound :: Bool
    True
    Prelude> minBound :: Bool
    False
    ```
  * ``Num``

    ```
    Prelude> :t 20
    20 :: Num a => a
    ```

    * Numerical typeclass.
    * Members can act like numbers.
  * ``Integeral``
    * Includes ``Int`` and ``Integer``
  * ``Floating``
    * Includes ``Float`` and ``Double``
    * Note: it appears this has been renamed ``Fractional`` in my version of Haskell:

      ```
      Prelude> :t 30.124
      30.124 :: Fractional a => a
      Prelude> :t 0.912
      0.912 :: Fractional a => a
      ```
* Use ``fromIntegral`` function to convert an integral number to general.
  * Useful for working with int and floats.

## Q & As

### Question: What are type variables?

Variables that can be assigned any types. Like "generics" in other languages. They are useful for function definitions where a function may accept multiple types, like ``succ``:

```
Prelude> :t succ
succ :: Enum a => a -> a
```

Here we can see that the ``succ`` function takes any type in type class ``Enum`` and returns a variable of the same type. Unlike a function like ``==`` that takes a variable in the typeclass ``Eq`` and returns a bool:

```
Prelude> :t (==)
(==) :: Eq a => a -> a -> Bool
```

***

### Question: Why are type classes useful?

For now, I know that they help define a range of types that a function can accept, where a function accepts a type variable.
