#include <stdio.h>

int main(int argc, char *argv[])
{
	int ages[] = {23, 43, 12, 89, 2};
	// Creating an array of strings
	char *names[] = {
		"Alan", "Frank", "Mary",
		"John", "Lisa"
	};

	// safely get size of ages
	int count = sizeof(ages) / sizeof(int);
	int i = 0;

	// first way using indexing
	for (i = 0; i < count; i++) {
		printf("%s has %d years alive.\n", names[i], ages[i]);
	}

	printf("---\n");

	// int * creates a 'pointer to integer' type pointer
	int *cur_age = ages;
	// char ** == a "pointer to (a pointer to char)" since names is 2d
	char **cur_name = names;


	// second way using pointers
	for (i = 0; i < count; i++) {
		printf("%s is %d years old.\n", *(cur_name + i), *(cur_age + i));
	}

	printf("---\n");

	// third way, pointers are just arrays
	for (i = 0; i < count; i++) {
		printf("%s is %d years old again.\n", cur_name[i], cur_age[i]);
	}

	printf("---\n");

	// fourth way with pointers in a complex way
	for (cur_name = names, cur_age = ages; (cur_age - ages) < count; cur_name++, cur_age++) {
		printf("%s lived %d years so far.\n", *cur_name, *cur_age);
	}

	return 0;
}
