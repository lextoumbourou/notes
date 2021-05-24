#include <stdio.h>
#include <stdlib.h>

struct Person {
	char *name;
};

void swap(struct Person ** p1, struct Person ** p2) {
	struct Person **temp = malloc(sizeof(struct Person));
	*temp = *p1;
	*p1 = *p2;
	*p2 = *temp;
}

int main() {
	struct Person *p1 = malloc(sizeof(struct Person));
	struct Person *p2 = malloc(sizeof(struct Person)); 

	p1->name = "John";
	p2->name = "Bob";

	printf("Name 1: %s\n", p1->name);
	printf("Name 2: %s\n", p2->name);

	swap(&p1, &p2);

	printf("Name 1: %s\n", p1->name);
	printf("Name 2: %s\n", p2->name);

	return 0;
}
