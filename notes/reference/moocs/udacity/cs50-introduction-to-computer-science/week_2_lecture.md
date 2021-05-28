Monday
======
###Line breaks

* Unix = \n
* Windows = \r\n (Closest typewriter representation:  scroll rolls, print head moves to left)
* Mac = \r

* To perform sudo animation in C, put carriage return in front of output with \r. This returns carriage to start of the line and wipes out the original output
(see source/percent_imp.py for example)

###Typecasting

* Processing of converting types eg int to char
char(65) = A
char(66) = B

* 65 to 65 + 26 is the capital letter alphabet
* 97 to 97 + 26 is the lower-case letter alphabet

Battleship
(see source/battleship.py)

Wednesday
========

## Hierarchical Decomposition

* Hierarchical (or functional) decomposition refers to splitting code into functions
* Declare != initialise variables
* Ternary operator can make code more elegant (or not)...

####C

```string s1 = (1 > 2) ? "yes" : "no"```

####Python

```'yes' if 2 > 1 else 'no'```

* In C, you must tell the compiler in advance that functions exist by initialising them otherwise you get Implicit Declaration of Function error.
* When functions are declared, you must specify their return value. For example, to specify a function without a return value:
```void chorus(int b);``` 

## Scope

* Pass in reference to variable to change in main scope.
```swap(int *a, int *b) {
    tmp = a;
    a = b;
    b = tmp;
}``` 

* Alternatively, you could declare the variable outside main() to make it a global variable. Doesn't scale very well, though.
* To swap variables in Python

```a = 1
b = 2
a, b = b, a```

* Buffer override is usually when a malicious program tries to write a value bigger then the compiler expects, and potential causes the computer to execute code it wouldn't otherwise. This can occur when a function calls another function, which places it higher in memory.
* Declare variable in function, pass it to another function. Function gets a copy of it because the second function's memory sits higher in the stack.
* If a function calls itself an infinite amount of time, you would probably get a segmentation fault.

## Strings

* Puts a "sentinal" value, at the end of string to represent end. Usually a null pointer (\0).
* A string, in C, is an array of chars, ending in \0
* You can access individual string values in C with array notation:
```printf("%c", str[1]);```
* Same in Python:
```print str[5]```

## Command line arguments

* Setup a arguments to main
```int
main (int argc, string argv[]) {
}```
* argc is number of arguments typed at the prompt
* argv is an array of arguments
* In Python:
```from sys import argv
print argv```

