# Syntax in functions

## Exercises

* Write a function that takes in two tuples of x, y coordinates and returns the following tuple: ```(slope, y-intercept)```.

## Pattern matching

* When defining functions, can separate function bodies for different patterns.

  ```
  eightHandler :: (Integral a) => a -> String
  eightHandler 8 = "You provided an eight."
  eightHandler x = "You provided something other than an 8. Sup with that?
  ```

  Example:

  ```
  ghci> eightHandler 8
  "You provided an eight."
  ghci> eightHandler 7
  "You provided something else."
  ```

* Example of factorial function:

  ```
  factorial :: (Integral a) => a -> a
  factorial 0 = 1
  factorial x = x * factorial (x - 1)
  ```

  In English:
    * **Line 1:** Take in a value of typeclass **Integral** and return a value of same typeclass.
    * **Line 2:** If function is called with a ``0``, return 1.
    * **Line 3:** If function is called with anything else, multiple the value by the result of the same function call with ```value - 1```

* If you define a function with non-exhaustive pattern, you get error as follows:

  ```
  ghci> let myFunc 'a' = "Alexis"
  ghci> let myFunc 'b' = "Bob"
  ghci> myFunc 'c'
  "*** Exception: <interactive>:16:5-22: Non-exhaustive patterns in function myFunc
  ```

* Can do patten matching in list comprehensions:

  ```
  ghci> let tups = [(1,3), (4,5), (2,6), (8,2)]
  ghci> [a + b | (a, b) <- tups, a > b]
  [10]
  ```

  English:
    * For each tuple in ```tups```, if the condition ```a > b``` is True, add ```a + b``` to new list.

* Lists can be used in pattern matching:

  Bind head of list to ```x``` and the rest of the list to ```xs```:

  ```
  ghci> let myHeadFunc (x:xs) = (x, xs)
  ghci> let xs = [(1,3), (4,5), (2,6), (8,2)]
  ghci> myHeadFunc xs
  ((1,3),[(4,5),(2,6),(8,2)])
  ```
  
  Note that list syntax ```[1, 2, 3]``` is syntactic sugar for ```1:2:3:[]``` (eh??).

* Use ```error``` function to generate a runtime error:

  ```
  ghci> let head' [] = error "Can't call head on empty list."
  ghci> let head' (x:_) = x
  ```

* Can use these patterns to count a list:

  ```
  sum' :: (Num a) => [a] -> a
  sum' [] = 0
  sum' (x:xs) = x + sum' xs
  ```

* Put a name and an ```@``` in front of a pattern to have access to whole pattern:

  ```
  firstLetter :: String -> String
  firstLetter whole@(x:xs) = "The first letter of " ++ whole ++ " is " ++ [x]
  ```

## Guards

* Equivalent of big if else or switch tree in "imperative" languages.

  Example:
  
  ```
  whatGeneration :: (Integral a) => a -> String
  whatGeneration yearOfBirth
      | 1925 < yearOfBirth && yearOfBirth <= 1945 = "You're the silent generation."
      | 1945 < yearOfBirth && yearOfBirth <= 1964 = "You're a baby boomer."
      | 1964 < yearOfBirth && yearOfBirth <= 1980 = "You're a Gen Xer."
      | 1980 < yearOfBirth && yearOfBirth <= 1999 = "You're a Gen Y."
      | 2000 < yearOfBirth = "You're a Millenial!"
      | otherwise = "No idea wtf you are, breh."
  ```

* Guard is a boolean expression after a pipe in a function declaration.
* Last guard is called ```otherwise```.
* Use ```where``` statement to create variables for guard conditions.

  Example:

  ```
  bmiTell :: (Integral a) => a -> String
  bmiTell weight heigh
    | bmi <= under  = "Underweight."
    | bmi <= normal = "Normal weight."
    | bmi <= fat    = "Overweight."
    | otherwise    = "Obese."
    where bmi = weight / height ^ 2
          skinny = 18.5
          normal = 25.0
          fat    = 30.0
  ```

  * var names are available only to that function.

## Let bindings

* ```let``` let's you bind variables and use them:

```
multipleBy8 x =
  let value = 8 in x * 8
```

* Can be used to introduce functions in local scopes.

## Case expressions

* Example:

```
head' :: [a] -> a
head' xs = case xs of [] -> error "No head for empty lists!"
                      (x:_) -> x
```
