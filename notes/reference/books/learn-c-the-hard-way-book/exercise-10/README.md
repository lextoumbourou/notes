# Exercise 10: Array of Strings, Looping

* the ```char *argv[]``` argument in the ```main``` function is an array of function arguments.
* The ```int argc``` is the length of arguments.
* ```argv[0]``` is the script name

```
for (i = 1; i < argc; i++) {
  printf("arg %d: %s\n", i, argv[i]);
}

```

* Syntax ```char *my_strs[]``` creates a multidimensional array, each string is 1 element, each char in the string is another.
