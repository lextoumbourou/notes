# Exercise 1: Dust Off That Compiler

* Simple program

```
int main(int argc, char *argv[])
{
    puts("Hello, world");

    return 0;
}
```

* To compile it:
```
> sudo apt-get install build-essentials
> make ex1
```

* To compile it with warnings:

```
> CFLAGS="-Wall" make ex1
cc -Wall    ex1.c   -o ex1
ex1.c:3:5: warning: implicit declaration of function 'puts' is invalid in C99 [-Wimplicit-function-declaration]
    puts("Hello, world");
    ^
1 warning generated.
```

* To fix the error, import the ``stdio.h`` header file, which defines a number of variables, functions and macros for performing i/o.

```
# include <stdio.h>

int main(int argc, char *argv[])
{
    puts("Hello, world");

    return 0;
}
```

* ``puts`` is used to write a string to stdout.
