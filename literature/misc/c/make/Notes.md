# Make

Notes from http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/.

* When running ``make`` with no arguments, it'll look for the first target.

```
> cat Makefile
sometarget: hellomake.c hellofunc.c
    gcc -o hellomake hellomake.c hellofunc.c -I.
> make
```

* Files put on the first line after ``sometarget:`` are used to tell make to only run the command when the files change.

* The constants ``CC`` and ``CFLAGS`` are known by ``make``:
  * ``CC`` - compiler to use.
  * ``CFLAGS`` - list of flags to pass in at compilation time.

```
> cat Makefile
CC=gcc
CFLAGS=-I.

sometarget: testmain.o testfunc.o
    gcc -o testmain testmain.o testfunc.o -I.
```


