#
# Makefile
#
# Computer Science 50
# Problem Set 5
#

all: copy resize whodunit

copy: bmp.h copy.c Makefile
	gcc -ggdb -std=c99 -Wall -Werror -o copy copy.c 

resize: bmp.h Makefile resize.c
	gcc -ggdb -std=c99 -Wall -Werror -o resize resize.c 

whodunit: bmp.h Makefile whodunit.c
	gcc -ggdb -std=c99 -Wall -Werror -o whodunit whodunit.c 

clean:
	rm -f *.o a.out copy core.* resize whodunit
