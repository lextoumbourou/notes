# Exercise 15: Pointers Dreaded Pointers

* What I think a pointer is:
	* An integer that represents a memory location where either some data or the beginning position of a sequence of data is stored.
* What pointers are
	* An array: ```int nums[] = {1, 2, 3, 4}``` is a chunk of memory set aside for storing values. It also represents an "address" of memory
		* When you "index" the array, you are accessing the value that far away from the array start, eg: ```nums[1]``` is 1-byte away from the start of the array
	* A pointer is simply another way of referring to this memory location
		* A pointer to array ```nums``` could be represented as: ```int cur_num = nums;```
	* You can "add" to a pointer like: ```*(cur_num + 1)``` this simply means: the address of the nums array + 1 byte.
	* Pointer lets you work with raw blocks of memory
* Practical Pointer Usage
	1. Works with chunks of memory like strings and structs
	2. Passing large blocks of memory (like large structs) to functions by reference
	3. Complex scanning of memory chunks ala converting bytes off network sockets or parsing files
* Pointer Lexicon
	* ```type *ptr``` - a pointer of type named ```ptr``` (eg ```int *my_ptr = some_array;```)
	* ```*ptr``` - value of whatever ```*ptr``` is pointed at
	* ```*(ptr + i)``` - value of (what ```ptr``` is pointed at + 1)
	* ```&something``` - address of ```something```
	* ```type *ptr = &something``` - a pointer of type name ptr set to address of ```something```
	* ```ptr++``` - increment where ptr points (I assume same as example 3)
