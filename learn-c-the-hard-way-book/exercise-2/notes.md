# Exercise 2: Make Is Your Python Now

* Pass environment variables to make (and other Unix progs)
* ```CFLAGS=-Wall``` will display warnings
```
> CFLAGS="-Wall" make ex1
```
* Basic ```Makefile``` to remove the bin file when ```make clean``` is run
```
> cat Makefile
CFLAGS=-Wall -g

clean:
    rm -f ex1
```
* To set a 'default' action, use the ```all``` directive
```
all:
    make ex1
```
