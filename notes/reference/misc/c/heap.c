#include <stdio.h>
#include <stdlib.h>

int main() {
	int a;  // goes on stack
	int *p;

	p = malloc(sizeof(int) * 2);

	// Memory can be dereferenced by using arrays
	p[0] = 1;
	p[1] = 2;
	printf("Numbers in memory: %d, %d", p[0], p[1]);

	// Or memory can be dereferences using points
	printf("First number: %d", *p);
	printf("Second number: %d", *(p + 1));

	free(p);

	return 0;
}
