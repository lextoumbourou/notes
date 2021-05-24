#include <stdio.h>
#include <assert.h> // provides assert() macro
#include <stdlib.h> // provides malloc()
#include <string.h> // provides strdup() function (amongst other things)

struct Person {
	char *name;
	int age;
	int height;
	int weight;
};

// function is used to create the struct Person
struct Person *Person_create(char *name, int age, int height, int weight)
{
	struct Person *who = malloc(sizeof(struct Person)); // request raw memory from the comp
	assert(who != NULL); // ensure memory was actually returned successfull

	who->name = strdup(name);
	who->age = age;
	who->height = height;
	who->weight = weight;

	return who;
}

void Person_destroy(struct Person *who)
{
	assert(who != NULL);

	free(who->name);
	free(who);
}

void Person_print(struct Person *who)
{
	printf("Name: %s\n", who->name);
	printf("\tAge: %d\n", who->age);
	printf("\tHeight: %d\n", who->height);
	printf("\tWeight: %d\n", who->weight);
}

int main(int argc, char *argv[])
{
	// make 2 people structures
	struct Person *lex = Person_create(
		"Lex T", 27, 180, 82);
	struct Person *bill = Person_create(
		"Bill B", 30, 182, 80);
	printf("Lex is at memory location %p:\n", lex);
	Person_print(lex);

	printf("Bill is at memory location %p:\n", bill);
	Person_print(bill);

	// make everyone age 20 years and print them out again
	lex->age += 20;
	lex->height -= 2;
	lex->weight += 40;
	Person_print(lex);

	bill->age += 20;
	bill->weight += 20;
	Person_print(bill);

	Person_destroy(lex);
	Person_destroy(bill);
}
