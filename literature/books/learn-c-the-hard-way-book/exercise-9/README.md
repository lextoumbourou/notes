# Exercise 9: Arrays and Strings

* In c, string == array of chars

* If you just set one element of an array in C, it'll fix the rest in with 0s.

```
int number[4] = {1}; // {1, 0, 0, 0}
```

* Though for char arrays, the 0s won't be printed as chaacters because they're treated as null terminators.

* These are two different ways to do a string:

```
char name[4] = {"L", "e", "x", "\0"};
char *name = "Lex";
```

The second is most common for string literals.
