# Exercise 2: Make Is Your Python Now

* Running ``make output_file`` does the following
  1. Does ``output_file`` already exist?
  2. If not, is there another file that starts with ``output_file``? Yes, ``output_file.c``.
  3. Do I know how to build ``.c`` files?
  4. Yes, run: ``cc output_file.c -o output_file``.

* Basic ```Makefile``` to remove the bin file when ```make clean``` is run:

  ```
  > cat Makefile
  CFLAGS=-Wall -g

  clean:
      rm -f ex1
  ```

* Setting the env var at the top of the Makefile means it'll be set each time ``make`` is run.

* To set a 'default' action, use the ```all``` directive:

  ```
  all: ex1
  ```

  Will run ``make ex1`` when no argument is passed to ``make``.

* More on Makefiles [here](../../../misc/c/make/Notes.md).
