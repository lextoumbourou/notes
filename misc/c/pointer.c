# include <stdio.h>

void add_one(int *number) {
	*number += 1;
}

int main() {
	int my_number = 1;

	printf("Number before func: %d\n", my_number);

	add_one(&my_number);

	printf("Number after func: %d\n", my_number);

	return 0;
}
