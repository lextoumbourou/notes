# Exercise 17: Heap and Stack Memory Allocation

* Heap
	* "All the remaining memory on your computer"
	* Everytime ```malloc``` is called, it allocated memory on the heap
	* When you call ```free``` you're freeing memory from the heap
* Stack
	* Special region of memory that stores temp variables available in a function's scope
	* When a function exits, C "pops" them off the stack to clean up
	* "If you didn't get it from ```malloc``` or a function that got it from ```malloc``` then it's on the stack
* 3 main problems to watch for
	* If you have a pointer to a block of memory from ```malloc``` and when the functions exits it'll get popped off and lost
	* Putting too much data on the stack (large structs and array) can cause a "stack overflow" and the program will crash. You should use the heap with malloc.
	* If you have a pointer to something on the stack, then pass or return it from your function, the function will "segmentation fault" (segfault) because the actual data will get popped off at function exit


