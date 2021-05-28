- in C, function has to return something

- Basic gcc usage:

gcc first.c

- To specify output with GCC:

gcc -o first first.c

- 'make' program works out whether to compile program or not, plus works out how to compile it
make first

- Typically an int can stored -2B to 2B or 4B

- if you want more, use bigger integers
- var types
int, float, long, string etc

- Boolean conditions:

AND

if(condition && condition){
}

OR
if(condition || condition){
}

- Switches:

switch(expression){
    case i:
        // do something
        break;

    case j:
        // do something
        break;

    default:
        // do something
        break;
}

