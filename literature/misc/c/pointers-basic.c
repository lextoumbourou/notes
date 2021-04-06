# include <stdio.h>

void add_one(int *value) {
	*value += 1;
}

int main() {
	int my_val = 5;

	printf("Val before function: %d\n", my_val);

	add_one(&my_val);

	printf("Val after function: %d\n", my_val);

	return 0;
}
