# Exercise 3: Formattted Printing

* Firstly, include header file, which includes "standard Input/Output function" like ```printf```
```
# include <stdio.h>
```
* Define variables, setting their type
```
int main()
{
    int age = 10;
    int height = 72;
```
* Use the ```printf``` function with ```%d``` format char to print height and age.
```
    printf('I am %d, years old\n', age);
    printf('I am %d inches tall\n', height);
```
* Return 0, so Unix knows the program completed succesfully
```
    return 0;
}
```

## printf format chars

* ```%d```, ```%i``` - int as a signed number (supports negative values)
* ```%u``` - unsigned int (non-negative)
* ```%f``` - double in "normal" notation
* ```%c``` - char (single character)
* ```%s``` - null-terminated string


